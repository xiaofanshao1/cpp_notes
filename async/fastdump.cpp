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

// ========== FastDump实现 ==========
FastDump::FastDump(const FastDumpConfig& config) : config_(config) {
    for (int i = 0; i < config_.surface_count; ++i) {
        surfaces_.emplace_back(new Surface(config_.width, config_.height, i));
        available_indices_.push_back(i);
    }
    if (config_.io_type == "file") {
        io_func_ = [this](const std::vector<Surface*>& surfaces) {
            make_dir_if_not_exists(config_.output_dir);
            for (auto* s : surfaces) {
                std::string fname = config_.output_dir + "/surface_" + std::to_string(s->id) + ".bin";
                std::ofstream ofs(fname, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(s->data.data()), s->data.size());
            }
        };
    } else if (config_.io_type == "mmap") {
        io_func_ = [this](const std::vector<Surface*>& surfaces) {
            make_dir_if_not_exists(config_.output_dir);
            for (auto* s : surfaces) {
                std::string fname = config_.output_dir + "/mmap_surface_" + std::to_string(s->id) + ".bin";
                std::ofstream ofs(fname, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(s->data.data()), s->data.size());
            }
        };
    } else {
        throw std::runtime_error("Unknown io_type in config");
    }
    io_future_ = io_promise_.get_future();
    stop_flag_ = false;
    scheduler_thread_ = std::thread(&FastDump::scheduler_thread_loop, this);
    io_thread_ = std::thread(&FastDump::io_thread_loop, this);
}

FastDump::FastDump() : FastDump(FastDumpConfig{}) {}

FastDump::~FastDump() {
    stop_flag_ = true;
    cv_main_.notify_all();
    cv_scheduler_.notify_all();
    io_cv_.notify_all();
    if (scheduler_thread_.joinable()) scheduler_thread_.join();
    if (io_thread_.joinable()) io_thread_.join();
}

void FastDump::scheduler_thread_loop() {
    while (!stop_flag_) {
        Surface* s = nullptr;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_scheduler_.wait(lock, [this] { return !ready_queue_.empty() || stop_flag_; });
            if (stop_flag_) break;
            s = ready_queue_.front();
            ready_queue_.pop();
        }
        // 放入IO batch
        {
            std::lock_guard<std::mutex> lock(io_mtx_);
            io_batch_.push_back(s);
            if (io_batch_.size() >= batch_size_) {
                io_cv_.notify_one();
            }
        }
        // 等待IO完成
        io_future_.wait();
        io_promise_ = std::promise<void>();
        io_future_ = io_promise_.get_future();

        // 回收surface
        s->is_written = true;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            available_indices_.push_back(s->id);
        }
        cv_main_.notify_one();
    }
}

void FastDump::io_thread_loop() {
    while (!stop_flag_) {
        std::vector<Surface*> batch;
        {
            std::unique_lock<std::mutex> lock(io_mtx_);
            io_cv_.wait(lock, [this] { return io_batch_.size() >= batch_size_ || stop_flag_; });
            if (stop_flag_) break;
            batch.swap(io_batch_);
        }
        if (!batch.empty()) {
            io_func_(batch);
            io_promise_.set_value();
        }
    }
}

void FastDump::dump(const Surface& src) {
    int idx = -1;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        if (available_indices_.empty()) {
            // 等待1ms，若还是没有则丢弃/返回
            if (!cv_main_.wait_for(lock, std::chrono::milliseconds(1), [this]{ return !available_indices_.empty(); })) {
                return;
            }
        }
        idx = available_indices_.back();
        available_indices_.pop_back();
    }
    // 拷贝数据到池 surface
    Surface* s = surfaces_[idx].get();
    s->data = src.data;
    s->is_ready = true;
    s->is_written = false;

    {
        std::lock_guard<std::mutex> lock(mtx_);
        ready_queue_.push(s);
    }
    cv_scheduler_.notify_one();
}