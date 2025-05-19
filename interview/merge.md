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

问题
1. 循环条件冗余：i<a.size()+b.size()在每次循环中都计算总和，效率低
2. 未预分配内存：res未提前分配a.size()+b.size()空间，导致多次扩容3

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

进一步