<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# LMDB 数据库

## 简介

- 记录在内存中的存储格式为 key-value 对

- 一条记录的数据格式为：data + label

- 采用内存－映射文件进行存储，读取速度快，支持多线程、多进程并发

- 常用于单标签分类问题

- 由于 Caffe 内部按照 b、g、r 三个通道分别存储图像，在生成 lmdb 数据时要调节矩阵维度

## Python 实现

### 写入

```
environment  = lmdb.open('train_data', map_size=1e12)

with environment.begin(write=True) as transaction:

	for idx, elem in enumerate(train_list):
		image = get_data(elem)
		image = numpy.transpose(image, (2, 0, 1))
		datum = caffe.proto.caffe_pb2.Datum()
		
		datum.channels = channel
		datum.height = height
		datum.width = width
		datum.data = image.tobytes()
		datum.label = label

		key = '%.8d' % idx
		transaction.put(key, datum.SerializeToString())
```

### 读取

```
environment = lmdb.open('train_data', readonly=True)
	
with environment.begin() as transaction:
	raw_datum = transaction.get('00000001')
		
datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

x = numpy.fromstring(datum.data, dtype=numpy.uint8)
data = x.reshape(datum.channels, datum.height, datum.width)
data = numpy.transpose(data, (1, 2, 0))
label = datum.label
```