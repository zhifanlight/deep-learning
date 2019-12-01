# 均值文件

## 简介

- 记录输入数据中每个像素点、每个通道的均值

- 对于 $M \times N$ 的彩色图像，$\mathrm{binaryproto}$ 文件维度为 $1 \times 3 \times M \times N$

## $\mathrm{Shell}$ 实现

### 写入

```shell
$CAFFE_ROOT/build/tools/compute_image_mean train_lmdb mean.binaryproto
```

## $\mathrm{Python}$ 实现

### 读取

```python
blob = caffe.proto.caffe_pb2.BlobProto()
binary_proto = open('mean.binaryproto', 'rb').read()
blob.ParseFromString(binary_proto)

array = numpy.array(caffe.io.blobproto_to_array(blob))
array = array[0]
```