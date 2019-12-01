# $\mathrm{SSD}$

## $\mathrm{train.prototxt}$

### 输入层

```protobuf
layer {
  name: "data"
  type: "AnnotatedData"
  top: "data"
  top: "label"
  transform_param {
    mirror: true
    mean_value: 104
    mean_value: 117
    mean_value: 123
    resize_param {
      # 先 distort、expand 再进行 resize
    }
    distort_param {
      # HSV 空间进行颜色变换
    }
    expand_param {
      # 将图像随机拷到一张（均值填充）大图上
    }
  }
  annotated_data_param {    # 从 batch_sampler 中随机选一种采样方式
    batch_sampler {
      max_sample: 1    # 从原图中采几个样本
      max_trials: 1    # 不满足采样条件时，最大尝试次数
    }
    ...
    batch_sampler {
      sampler {    # 随机缩放、拉伸，同时也更新对应的 Ground Truth
        min_scale: 0.3           # 默认 1.0
        max_scale: 1.0           # 默认 1.0
        min_aspect_ratio: 0.5    # 默认 1.0
        max_aspect_ratio: 2.0    # 默认 1.0
      }
      sample_constraint {
        min_jaccard_overlap: th    # 采样条件：IoU > th
      }
      max_sample: 1
      max_trials: 50
    }
  }
}
```

### $\mathrm{BBox}$ 输出

```protobuf
layer {
  name: "conv4_3_norm_mbox_loc"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_loc"
  convolution_param {
    num_output: 16    # 4 种 default box，每种对应 4 个变换量
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_loc"
  top: "conv4_3_norm_mbox_loc_perm"
  permute_param {    # 将 NCHW 变换为 NHWC，方便下一步的 flatten
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_loc_perm"
  top: "conv4_3_norm_mbox_loc_flat"
  flatten_param {    # 将数据拉成二维，方便 concat 后统一处理
    axis: 1
  }
}
```

### $\mathrm{Softmax}$ 输出

```protobuf
layer {
  name: "conv4_3_norm_mbox_conf"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_conf"
  convolution_param {
    num_output: 84    # 21 类，4 种 default box 的置信度
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_conf"
  top: "conv4_3_norm_mbox_conf_perm"
  permute_param {    # 将 NCHW 变换为 NHWC，方便下一步的 flatten
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_conf_perm"
  top: "conv4_3_norm_mbox_conf_flat"
  flatten_param {    # 将数据拉成二维，方便 concat 后统一处理
    axis: 1
  }
}
```

### $\mathrm{PriorBox}$ 层

```protobuf
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
  prior_box_param {
    min_size: 30.0     # default box 对应原图的最小像素值
    max_size: 60.0     # default box 对应原图的最大像素值
    aspect_ratio: 2    # 除了 1:1，再使用 1: 2 的长宽比
    flip: true         # 使用相对应的宽长比
    clip: false        # 不把超出边界的 default box 截断
    variance: 0.1      # 扩大 BBox 的 4 个变换量，加速收敛
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 8            # 经过了 3 次 2 x 2 pooling
    offset: 0.5        # 特征图上中心点的偏移量，类似双线性插值的中心对齐
  }
}
```

- $\mathrm{PriorBox}$ 层处理流程：

  - 生成 $\mathrm{ratio} = 1:1, \quad \mathrm{height = min, \quad width = min}$ 的 $\mathrm{Default \ Box}$

  - 生成 $\mathrm{ratio} = 1:1, \quad \mathrm{height = \sqrt{min \cdot max}, \quad width = \sqrt{min \cdot max}}$ 的 $\mathrm{Default \ Box}$

  - 根据 `aspect_ratio` 和 `flip` 参数，生成其他的 $\mathrm{Default \ Box}$

  - 记录 `variance` ，用于计算 $\mathrm{loss}$

### $\mathrm{Concat}$ 层

```protobuf
layer {    # 拼接不同特征图的 bbox
  name: "mbox_loc"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_loc_flat"
  ...
  bottom: "conv9_2_mbox_loc_flat"
  top: "mbox_loc"
  concat_param {
    axis: 1
  }
}
layer {    # 拼接不同特征图的 softmax
  name: "mbox_conf"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_conf_flat"
  ...
  bottom: "conv9_2_mbox_conf_flat"
  top: "mbox_conf"
  concat_param {
    axis: 1
  }
}
layer {    # 拼接不同特征图的 default box
  name: "mbox_priorbox"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_priorbox"
  ...
  bottom: "conv9_2_mbox_priorbox"
  top: "mbox_priorbox"
  concat_param {
    axis: 2
  }
}
```

### $\mathrm{MultiBoxLoss}$ 层

```protobuf
layer {
  name: "mbox_loss"
  type: "MultiBoxLoss"
  bottom: "mbox_loc"         # bbox
  bottom: "mbox_conf"        # softmax
  bottom: "mbox_priorbox"    # default box
  bottom: "label"            # ground truth
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1     # 回归 loss 使用 smooth L1
    conf_loss_type: SOFTMAX      # 分类 loss 使用 softmax
    loc_weight: 1.0              # 回归 loss 权重
    num_classes: 21              # 分类的类别数
    share_location: true         # 不同类共享 bbox 参数
    match_type: PER_PREDICTION
    overlap_threshold: 0.5       # 正样本 IoU 阈值
    background_label_id: 0       # 背景类 id
    use_difficult_gt: true
    neg_pos_ratio: 3.0           # 负样本、正样本比例
    neg_overlap: 0.5             # 小于此阈值视为负样本
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE    # 根据 score 选择负样本
  }
}
```

- $\mathrm{MultiBoxLoss}$ 层处理流程：

  - 获取所有 $\mathrm{Ground \ Truth}$、$\mathrm{Default \ Box}$、$\mathrm{BBox}$ 信息

  - 对 $\mathrm{Ground \ Truth}$ 和 $\mathrm{Default \ Box}$ 进行匹配

    - 根据匹配结果，计算 $\mathrm{BBox}$ 目标

  - 进行 $\mathrm{Hard \ Negative \ Mining}$

    - 计算分类损失

## $\mathrm{deploy.prototxt}$

### 输入层

```protobuf
input: "data"    # 待检测图像
input_shape {
  dim: 1
  dim: 3
  dim: 300
  dim: 300
}
```

### $\mathrm{Softmax}$ 层

```protobuf
layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "mbox_conf"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0     # 维度不变
      dim: -1    # 根据其他维度计算
      dim: 21    # 21 分类
    }
  }
}
layer {
  name: "mbox_conf_softmax"
  type: "Softmax"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_softmax"
  softmax_param {    # 按类别进行 softmax
    axis: 2
  }
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_softmax"
  top: "mbox_conf_flatten"
  flatten_param {    # 将数据拉成二维，方便后续处理
    axis: 1
  }
}
```

### $\mathrm{DetectionOutput}$ 层

```protobuf
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc"             # bbox
  bottom: "mbox_conf_flatten"    # softmax
  bottom: "mbox_priorbox"        # default box
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 21           # 21 分类
    share_location: true      # 不同类共享 bbox 参数
    background_label_id: 0    # 背景类 id
    nms_param {
      nms_threshold: 0.45     # nms 阈值
      top_k: 400              # 保留置信度最高的 400 个样本进行 nms
    }
    save_output_param {
      output_directory: path                     # 输出文件保存路径
      output_name_prefix: prefix                 # 文件前缀
      output_format: "VOC"
      label_map_file: "labelmap_voc.prototxt"    # 类映射文件
      name_size_file: "test_name_size.txt"       # 待测试图片信息（name height width）
      num_test_image: 4952                       # 待测试图片数量
    }
    code_type: CENTER_SIZE
    keep_top_k: 200               # nms 后，最多保留 200 个检测目标
    confidence_threshold: 0.01    # 最低置信度
  }
}
```

- $\mathrm{DetectionOutput}$ 层处理流程：

  - 根据 $\mathrm{Default \ Box}$ 和 $\mathrm{BBox}$ 变换量，得到原图上的目标位置（百分比）

  - 保留置信度最高的 `top_k` 个目标，之后进行 $\mathrm{NMS}$

  - 如果目标过多，保留置信度最高的 `keep_top_k` 个目标

## 网络输出

- $\mathrm{BBox}$ 输出 `mbox_loc` 维度为 $\left( 1, \ 4D \right)$

  - $D$ 表示所有特征图上所有 $\mathrm{Default \ Box}$ 数量

- $\mathrm{Softmax}$ 输出 `mbox_conf_flatten` 维度为 $\left( 1, \ DC \right)$

  - $C$ 表示分类的类别数

- $\mathrm{PriorBox}$ 层输出 `mbox_priorbox` 维度为 $\left( 1, \ 2, \ 4D \right)$

  - 第一个通道存储每个 $\mathrm{Default \ Box}$ 在原图上的位置 $\mathrm{x1, \ y1, \ x2, \ y2}$

  - 第二个通道存储对应的 `variance`，用于计算 $\mathrm{loss}$

- 最终输出 `detection_out` 维度为 $\left( 1, \ 1, \ K, \ 7 \right)$

  - $K$ 是每张图上检测到的目标数量

  - $7$ 维向量分别是 $\mathrm{id, \ class, \ conf, \ x1, \ y1, \ x2, \ y2}$

    - $\mathrm{id}$ 表示当前图像在 $\mathrm{batch}$ 内的序号

    - $\mathrm{class}$ 表示预测的分类

    - $\mathrm{conf}$ 表示对应的置信度

- 设置新阈值，过滤掉 `detection_out` 的大部分结果，输出检测到的目标

- $\mathrm{SSD}$ 使用的 $6$ 个特征图尺度依次为：

  $$
  38, \ 19, \ 10, \ 5, \ 3, \ 1
  $$

- $\mathrm{MobileNet-SSD}$ 使用的 $6$ 个特征图尺度依次为：

  $$
  19, \ 10, \ 5, \ 3, \ 2, \ 1
  $$