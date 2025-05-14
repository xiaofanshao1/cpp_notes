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