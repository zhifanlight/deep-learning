# $\mathrm{LMDB}$ 数据库

## 简介

- 记录在内存中的存储格式为 $\mathrm{key-value}$ 对

- 一条记录的数据格式为：$\mathrm{data + label}$

- 采用内存－映射文件进行存储，读取速度快，支持多线程、多进程并发

- 常用于单标签分类问题

- $\mathrm{Caffe}$ 内部按 $\mathrm{B, \ G, \ R}$ 三个通道分别存储数据，需要对图像进行维度变换

## $\mathrm{Shell}$ 实现

### 写入

```shell
DATA='inputs'

TRAIN='inputs/train/'
VAL='inputs/val/'

HEIGHT=0
WIDTH=0

GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset --resize_height=$HEIGHT --resize_width=$WIDTH --shuffle $VAL $DATA/val.txt $DATA/val_lmdb

GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset --resize_height=$HEIGHT --resize_width=$WIDTH --shuffle $TRAIN $DATA/train.txt $DATA/train_lmdb
```

## $\mathrm{Python}$ 实现

### 写入

```python
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

#### 读取指定图片

```python
environment = lmdb.open('train_data', readonly=True)
datum = caffe.proto.caffe_pb2.Datum()

with environment.begin() as transaction:
  raw_datum = transaction.get('00000001')
  datum.ParseFromString(raw_datum)
  label = datum.label
  data = caffe.io.datum_to_array(datum)
  image = data.transpose(1, 2, 0)
  cv2.imwrite('lmdb.jpg', image)
```

#### 读取所有图片

```python
environment = lmdb.open('train_data', readonly=True)
datum = caffe.proto.caffe_pb2.Datum()

with environment.begin() as transaction:
  for key, value in transaction.cursor():
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    image = data.transpose(1, 2, 0)
    cv2.imwrite('lmdb.jpg', image)
```