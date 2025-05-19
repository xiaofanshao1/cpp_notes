# 场景: 有序归并

两个有序数组合并

```cpp
vector<int> mergeArray(vector<int> &a, vecotr<int> &b){
    vector<int> res;
    for(int i=0,u=0,v=0;i<a.size()+b.size();i++){
      if(u<a.size() && v < b.size()){
        if(a[u]<b[v]){
          res.emplace_back(a[u]);
          u++;
         } 
        else {
          res.emplace_back(b[v]);
          v++;
        }else{break;
        }
     } 

      if(u==a.size()) {
        while(v<b.size())res.emplace_back(b[v++]);
        }else if(v==b.size(){
        while(u<a.size())res.emplace_back(a[u++]);
        }

    return res;
}
```

**问题**

1. 循环条件冗余：i<a.size()+b.size()在每次循环中都计算总和，效率低
2. 未预分配内存：res未提前分配a.size()+b.size()空间，导致多次扩容3

```cpp
vector<int> mergeArray(const vector<int>& a, const vector<int>& b) {
  vector<int> res;
  res.reserve(a.size() + b.size());  // 预分配避免扩容开销

  size_t u = 0, v = 0;
  while (u < a.size() && v < b.size()) {
      res.push_back(a[u] < b[v] ? a[u++] : b[v++]);  // 三目运算简化
  }

  // 直接插入剩余元素（无需循环判断）
  res.insert(res.end(), a.begin() + u, a.end());
  res.insert(res.end(), b.begin() + v, b.end());

  return res;
}
```

**如果拓展到两个大文件**

- 流式读取+缓冲区块    写 IO
- 可以做的优化
  - 缓冲区大小调节
  - double buffer  prefetch
  - 持久化处理 防止 failure
  - mmap 读写
  - 当文件进一步增大时，可以分开多个有序段

```cpp
#include <fstream>
#include <vector>
#include <algorithm> // min

class StreamMerger {
    static constexpr size_t BUFFER_SIZE = 1 << 20; // 1M

    struct FileStream {
        std::ifstream file;
        std::vector<int> buffer;
        size_t pos = 0;
        
        explicit FileStream(const std::string& path) : file(path, std::ios::binary) {}
        
        bool load() {
            buffer.resize(BUFFER_SIZE);
            file.read(reinterpret_cast<char*>(buffer.data()), BUFFER_SIZE*sizeof(int));
            buffer.resize(file.gcount()/sizeof(int));
            pos = 0;
            return !buffer.empty();
        }
    };

    FileStream in1, in2;
    std::ofstream out;

public:
    StreamMerger(const std::string& f1, const std::string& f2, const std::string& dst)
    : in1(f1), in2(f2), out(dst, std::ios::binary) 
    {
        in1.load();
        in2.load();
    }

    void merge() {
        auto write_remain = [this](FileStream& fs) {
            while(fs.pos < fs.buffer.size() || fs.load()) {
                auto* data = fs.buffer.data() + fs.pos;
                auto size = (fs.buffer.size() - fs.pos) * sizeof(int);
                out.write(reinterpret_cast<char*>(data), size);
                fs.pos = fs.buffer.size();
            }
        };
        
        while(in1.pos < in1.buffer.size() && in2.pos < in2.buffer.size()) {
            auto& cur = in1.buffer[in1.pos] < in2.buffer[in2.pos] ? in1 : in2;
            out.write(reinterpret_cast<char*>(&cur.buffer[cur.pos++]), sizeof(int));
            
            if(cur.pos == cur.buffer.size() && !cur.load()) { 
                (cur.buffer.data() == in1.buffer.data() ? write_remain(in2) : write_remain(in1));
                return;
            }
        }
    }
};

// 使用示例
int main() {
    StreamMerger("data1.bin", "data2.bin", "merged.bin").merge();
}
```

