<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 内存同步

## 实现

- 通过 ```SyncedMemory``` 类实现 GPU 和 CPU 之间的数据同步

	- ```SyncedHead``` 枚举类记录上一次的数据状态

	- ```gpu_ptr_``` 和 ```cpu_ptr_``` 分别是 GPU 和 CPU 数据指针

- 可能需要数据同步的 API 如下：

	- ```gpu_data()```：返回 GPU 上的常量指针，用于 GPU 数据读取

	- ```cpu_data()```：返回 CPU 上的常量指针，用于 CPU 数据读取

	- ```mutable_gpu_data()```：返回 GPU 上的普通指针，用于 GPU 数据写入

	- ```mutable_cpu_data()```：返回 CPU 上的普通指针，用于 CPU 数据写入

- 操作 GPU 指针时，会调用 ```to_gpu()```；操作 CPU 指针时，会调用 ```to_cpu()```

### to_gpu()

- 如果 ```head_``` 为 ```HEAD_AT_CPU```，进行数据拷贝，并将 ```head_``` 更新为 ```SYNCED```

### to_cpu()

- 如果 ```head_``` 为 ```HEAD_AT_GPU```，进行数据拷贝，并将 ```head_``` 更新为 ```SYNCED```

### mutable\_data()

- 调用完 ```mutable_gpu_data()``` 后，```head_``` 更新为 ```HEAD_AT_GPU```

- 调用完 ```mutable_cpu_data()``` 后，```head_``` 更新为 ```HEAD_AT_CPU```