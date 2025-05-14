# FastDump: 高性能异步视频帧 Dump 管理

### 项目简介

`FastDump` 是一个基于 C++11 标准库实现的高性能异步视频帧 dump 管理类。它采用多线程、条件变量、future/promise 机制，实现了主线程、调度线程、IO线程的生产-消费-调度异步协作，适用于高吞吐量的视频帧异步写盘场景。

### 设计思路

1. 主线程和 dump 过程异步，不降低主线程任务吞吐量

2. 利用 GPU 显存池，主线程encode 掩盖 surface copy 产生的延迟

3. IO 速度一定是比 GPU surface copy 速度要慢的，服务好 IO ，让 IO 线程满载读写很重要。

![plantuml-diagram-1747232450549](./assets/plantuml-diagram-1747232450549.svg)

### 实现细节

- **C++11 多线程与同步**：仅依赖标准库的 thread、mutex、condition_variable、future。
- **主线程接口极简**：主线程只需生成 surface 并调用 `dump`，无需关心池、同步、调度等细节。
- **Surface 池自动管理**：内部维护 surface 池和ready队列，自动分配与回收。
- **调度与 IO 分离**：专职调度线程处理同步逻辑，与IO 线程的解耦，IO 线程专职批量写盘。
- **多个 IO 策略**：通过 lambda + std::function 实现 IO 策略多态，支持普通文件写和 mmap 写，策略由 config 配置。
- **高效批量写盘**：支持 surface 批量写盘，减少磁盘寻道，提升吞吐

### 用法

```cpp
#include "fastdump.h"
#include <thread>
#include <iostream>

int main() {
    FastDump fastdump;

    int width = 1920, height = 1080;
    std::thread main_thread([&fastdump, width, height] {
        while (true) {
            Surface tmp(width, height, -1);
            std::generate(tmp.data.begin(), tmp.data.end(), []{ return rand() % 256; });
            fastdump.dump(tmp);
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(5));
    main_thread.join();

    std::cout << "FastDump finished." << std::endl;
    return 0;
}
```

## 配置说明

```cpp
struct FastDumpConfig {
    std::string io_type = "file";      // "file" 或 "mmap"
    std::string output_dir = "./dump";
    int surface_count = 50;
    int width = 1920;
    int height = 1080;
    int batch_size = 8;
};
```

``` plantuml
@startuml
!theme plain
participant "主线程" as Main
participant "调度线程" as Scheduler
participant "IO线程" as IO

box "同步资源池" #F5F5F5
participant "Surface池" as Pool
participant "ReadyQueue" as ReadyQueue
end box

== 主线程操作流 ==

Main -> Pool++: 1. 加锁取Surface
Pool --> Main--: 弹出空闲Surface

Main -> Main: 2. 填充数据\n（无锁操作）
Main -> ReadyQueue++: 3. 加锁入队就绪任务
ReadyQueue --> Main-- 
Main -> Scheduler: 4. notify_condition
activate Scheduler #lightgreen

== 并行事件处理 ==

IO --> Scheduler: 5. IO完成回调\n(notify_condition)
activate Scheduler #lightcoral

|||
group 调度线程的临界区 [双锁保护]
Scheduler -> Scheduler: 6. 同步处理两种事件：
note right
- 事件类型A：处理就绪队列（需队列锁）
- 事件类型B：Surface回收（需池锁）
end note

alt 处理就绪队列 [获取队列锁]
 Scheduler -> ReadyQueue++: 7a. 锁定就绪队列
 ReadyQueue --> Scheduler: 取出待处理任务
 Scheduler -> ReadyQueue--: 释放队列锁
 Scheduler -> IO++: 8a. 启动异步IO
 IO --> Scheduler--
else 处理资源回收 [获取池锁]
 Scheduler -> Pool++: 7b. 锁定Surface池
 Pool -> Pool: 插入可用列表
 Pool --> Scheduler--
end
end group
|||

deactivate Scheduler
@enduml

```

