#include "fastdump.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>

// 仅支持 macOS/Linux 的目录创建函数
static bool make_dir_if_not_exists(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) != 0) {
        return mkdir(dir.c_str(), 0755) == 0;
    }
    return true;
}

FastDump::FastDump(const FastDumpConfig& config) : config_(config) {
    // 初始化 Surface 池
    for (int i = 0; i < config_.surface_count; ++i) {
        surfaces_.emplace_back(new Surface(config_.width, config_.height, i));
        available_indices_.push_back(i);
    }
    
    // 初始化 IO 函数
    if (config_.io_type == "file") {
        io_func_ = [this](Surface* s) {
            make_dir_if_not_exists(config_.output_dir);
            std::string fname = config_.output_dir + "/surface_" + std::to_string(s->id) + ".bin";
            std::ofstream ofs(fname, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(s->data.data()), s->data.size());
        };
    } else if (config_.io_type == "mmap") {
        io_func_ = [this](Surface* s) {
            make_dir_if_not_exists(config_.output_dir);
            std::string fname = config_.output_dir + "/mmap_surface_" + std::to_string(s->id) + ".bin";
            std::ofstream ofs(fname, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(s->data.data()), s->data.size());
        };
    } else {
        throw std::runtime_error("Unknown io_type in config");
    }
    
    // 启动调度线程
    stop_flag_ = false;
    scheduler_thread_ = std::thread(&FastDump::scheduler_thread_loop, this);
}

FastDump::FastDump() : FastDump(FastDumpConfig{}) {}

FastDump::~FastDump() {
    // 发送关闭事件并设置停止标志
    {
        std::lock_guard<std::mutex> lock(event_mtx_);
        event_queue_.push(Event(EventType::SHUTDOWN, nullptr));
        stop_flag_ = true;
    }
    
    // 通知所有等待线程
    event_cv_.notify_one();
    cv_main_.notify_all();
    
    // 等待调度线程结束
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
    
    // 等待所有进行中的任务完成（这是必要的，确保没有悬空的线程）
    std::lock_guard<std::mutex> lock(tasks_mtx_);
    for (auto& task : pending_io_tasks_) {
        if (task.second.valid()) {
            task.second.wait();
        }
    }
}

std::future<void> FastDump::process_surface_async(Surface* surface) {
    // 创建一个packaged_task封装IO操作和完成后的回调
    auto task = std::make_shared<std::packaged_task<void()>>([this, surface]() {
        // 执行IO操作
        io_func_(surface);
        
        // IO完成后，发送事件到事件队列
        {
            std::lock_guard<std::mutex> lock(event_mtx_);
            if (!stop_flag_) { // 只有在未停止时才发送事件
                event_queue_.push(Event(EventType::IO_COMPLETED, surface));
                event_cv_.notify_one();
            }
        }
    });
    
    // 获取future
    std::future<void> future = task->get_future();
    
    // 启动任务
    std::thread([task]() { (*task)(); }).detach();
    
    return future;
}

void FastDump::on_io_completed(Surface* surface) {
    if (!surface) return;
    
    // 标记surface为已写入并回收
    surface->is_written = true;
    
    // 移除任务记录
    {
        std::lock_guard<std::mutex> lock(tasks_mtx_);
        pending_io_tasks_.erase(surface);
    }
    
    // 回收surface到可用池
    {
        std::lock_guard<std::mutex> lock(pool_mtx_);
        available_indices_.push_back(surface->id);
        cv_main_.notify_one(); // 通知可能等待的dump线程
    }
}

void FastDump::scheduler_thread_loop() {
    while (!stop_flag_) {
        Event event;
        bool has_event = false;
        
        // 等待事件
        {
            std::unique_lock<std::mutex> lock(event_mtx_);
            event_cv_.wait(lock, [this] { 
                return !event_queue_.empty() || stop_flag_; 
            });
            
            if (!event_queue_.empty()) {
                event = event_queue_.front();
                event_queue_.pop();
                has_event = true;
            } else if (stop_flag_) {
                break;
            }
        }
        
        if (!has_event) continue;
        
        // 处理事件
        switch (event.type) {
            case EventType::NEW_SURFACE:
                // 启动异步IO并跟踪任务
                {
                    std::lock_guard<std::mutex> lock(tasks_mtx_);
                    pending_io_tasks_[event.surface] = process_surface_async(event.surface);
                }
                break;
                
            case EventType::IO_COMPLETED:
                // 处理IO完成，回收surface
                on_io_completed(event.surface);
                break;
                
            case EventType::SHUTDOWN:
                // 直接返回，结束线程
                return;
        }
    }
}

void FastDump::dump(const Surface& src) {
    // 获取可用surface
    int idx = -1;
    {
        std::unique_lock<std::mutex> lock(pool_mtx_);
        if (available_indices_.empty()) {
            // 等待1ms，若还是没有则丢弃/返回
            if (!cv_main_.wait_for(lock, std::chrono::milliseconds(1), 
                 [this]{ return !available_indices_.empty(); })) {
                return;
            }
        }
        idx = available_indices_.back();
        available_indices_.pop_back();
    }
    
    if (idx < 0 || idx >= surfaces_.size()) return; // 安全检查
    
    // 拷贝数据到池surface
    Surface* s = surfaces_[idx].get();
    s->data = src.data;
    s->is_ready = true;
    s->is_written = false;

    // 加入事件队列并通知调度线程
    {
        std::lock_guard<std::mutex> lock(event_mtx_);
        if (!stop_flag_) { // 只有在未停止时才发送事件
            event_queue_.push(Event(EventType::NEW_SURFACE, s));
            event_cv_.notify_one();
        } else {
            // 系统正在关闭，释放surface
            std::lock_guard<std::mutex> pool_lock(pool_mtx_);
            available_indices_.push_back(idx);
        }
    }
}