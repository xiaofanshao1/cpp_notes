
# 生成 fastdump_main 可执行文件
add_executable(fastdump_main fastdump_main.cpp fastdump.cpp)
target_include_directories(fastdump_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) 