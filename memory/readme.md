# Memory allocation



### 问题：交替分配混合尺寸块产生碎片





### 混合策略：伙伴系统 + 内存池

对高频分配的固定大小（如 250KB）启用专用内存池，对其他大小使用伙伴系统；多层次 Memory pool可避免锁占用。small buffer pool：一般情况下是对于小于内存页 4KB 的，真实场景需要结合业务



- 内部碎片：内存对齐 256KB

- 问题

  - 如何进行地址的 `4K`对齐？

    好处是用来合并访存

    ```cpp
    //cuda 例子
    size_t alignment = 4096;
    size_t size = 8192;
    void* d_ptr;
    cudaMalloc(&d_ptr, size + alignment - 1);
    void* aligned_d_ptr = (void*)( ( (size_t)d_ptr + alignment - 1 ) & ~(alignment - 1) );
    // 验证对齐
    if ( (size_t)aligned_d_ptr % alignment != 0 ) {
        std::cerr << "Alignment failed!" << std::endl;
    } else {
        std::cout << "4K-aligned address: " << aligned_d_ptr << std::endl;
    }
    cudaFree(d_ptr); // 释放原始指针
    ```

    

  - 交替大小尺寸的循环申请显存，在clCreateBuffer 会消耗时间变长到1ms 为什么？

    独显`CL_MEM_USE_HOST_PTR `-> `CL_MEM_COPY_HOST_PTR` 是两部分时间：内存分配+内存拷贝。前者碎片化，后者跨越 PCIE。

    优化方式

    - clCreateBuffer 通过 small buffer pool可以 O1 时间获得显存

    - 对于小 buffer，在 driver 当中计算 hash，通过 hash+LRU 的方式在省去了独显时频繁穿越 PCIE 造成的 copy