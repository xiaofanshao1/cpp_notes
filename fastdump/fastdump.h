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
    int batch_size = 8;
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
    void io_thread_loop();

    FastDumpConfig config_;
    std::function<void(const std::vector<Surface*>&)> io_func_;

    // surface池
    std::vector<std::unique_ptr<Surface>> surfaces_;
    std::vector<int> available_indices_;
    std::queue<Surface*> ready_queue_;

    // 线程与同步
    std::mutex mtx_;
    std::condition_variable cv_main_;
    std::condition_variable cv_scheduler_;

    // IO线程相关
    std::mutex io_mtx_;
    std::condition_variable io_cv_;
    std::vector<Surface*> io_batch_;
    std::promise<void> io_promise_;
    std::future<void> io_future_;

    // 线程
    std::thread scheduler_thread_;
    std::thread io_thread_;
    std::atomic<bool> stop_flag_{false};
};