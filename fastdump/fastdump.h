#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <string>
#include <atomic>
#include <cstdint>
#include <functional>
#include <unordered_map>

// Surface结构体
struct Surface {
    std::vector<uint8_t> data;
    int width;
    int height;
    int id;
    bool is_ready = false;
    bool is_written = false;
    Surface(int w, int h, int i) : data(w * h), width(w), height(h), id(i) {}
};

// 配置结构体
struct FastDumpConfig {
    std::string io_type = "file";      // "file" 或 "mmap"
    std::string output_dir = "./dump";
    int surface_count = 50;
    int width = 1920;
    int height = 1080;
};

// 事件类型
enum class EventType {
    NEW_SURFACE,    // 新的Surface需要处理
    IO_COMPLETED,   // IO操作完成
    SHUTDOWN        // 系统关闭
};

// 事件结构体
struct Event {
    EventType type;
    Surface* surface;
    
    Event() : type(EventType::SHUTDOWN), surface(nullptr) {}
    Event(EventType t, Surface* s) : type(t), surface(s) {}
};

// FastDump主类
class FastDump {
public:
    FastDump();
    FastDump(const FastDumpConfig& config);
    ~FastDump();

    void dump(const Surface& src);

private:
    void scheduler_thread_loop();
    std::future<void> process_surface_async(Surface* surface);
    void on_io_completed(Surface* surface);

    // 配置
    FastDumpConfig config_;
    std::function<void(Surface*)> io_func_;

    // surface池
    std::vector<std::unique_ptr<Surface>> surfaces_;
    std::vector<int> available_indices_;
    std::mutex pool_mtx_;
    std::condition_variable cv_main_;

    // 事件系统
    std::queue<Event> event_queue_;
    std::mutex event_mtx_;
    std::condition_variable event_cv_;
    
    // 进行中的IO任务跟踪
    std::unordered_map<Surface*, std::future<void>> pending_io_tasks_;
    std::mutex tasks_mtx_;

    // 线程控制
    std::thread scheduler_thread_;
    std::atomic<bool> stop_flag_{false};
};