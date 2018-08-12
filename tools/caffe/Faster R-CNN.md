<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Faster R-CNN

## deploy.prototxt

### 输入层

```
input: "data"			# 待检测图像
input_shape {
	dim: 1
	dim: 3
	dim: 224
	dim: 224
}

input: "im_info"		# batch 数量、通道数
input_shape {
	dim: 1
	dim: 3
}
```

### Proposal 层

```
layer {
	name: 'proposal'
	type: 'Python'
	bottom: 'rpn_cls_prob_reshape'		# RPN softmax 输出
	bottom: 'rpn_bbox_pred'				# RPN bbox 输出
	bottom: 'im_info'					# 输入 shape
	top: 'rois'							# Region Proposal
	python_param {
		module: 'rpn.proposal_layer'
		layer: 'ProposalLayer'
		param_str: "'feat_stride': 16"	# 经过了 4 次 2 x 2 pooling
	}
}
```

- Proposal 层处理过程：

	- 根据 RPN 的 bbox 结果，得到输入图像上的 Region Proposal

	- 超过图像边界的 Region Proposal，直接截断到图像边界

	- 删除边长小于 ```RPN_MIN_SIZE``` 的小尺度目标

	- 跟据 RPN 的 softmax 结果进行排序，保留得分最高的 ```RPN_PRE_NMS_TOP_N``` 个 Region Proposal

	- 进行 NMS，返回得分最高的 ```RPN_POST_NMS_TOP_N``` 个 Region Proposal

### ROIPooling 层

```
layer {
	name: "roi_pool5"
	type: "ROIPooling"
	bottom: "conv5_3"		# 卷积层特征图
	bottom: "rois"			# 得到的 Region Proposal
	top: "pool5"			# roi pooling 结果
	roi_pooling_param {
		pooled_w: 7		# 输出特征图的宽
		pooled_h: 7		# 输出特征图的高
		spatial_scale: 0.0625	# VGG 4 次 pooling
	}
}
```

### Fast R-CNN Softmax 层

```
layer {
	name: "cls_score"
	type: "InnerProduct"
	bottom: "fc7"
	top: "cls_score"
	inner_product_param {
		num_output: 21		# 背景 + 20 类目标
	}
}

layer {
	name: "cls_prob"		# softmax 结果
	type: "Softmax"
	bottom: "cls_score"
	top: "cls_prob"
}
```

### Fast R-CNN BBox 层

```
layer {
	name: "bbox_pred"
	type: "InnerProduct"
	bottom: "fc7"
	top: "bbox_pred"
	inner_product_param {
		num_output: 84		# 背景 + 20 类目标，每类 4 个 bbox 变换量
	}
}
```

## train.prototxt

### 输入层

```
layer {
	name: 'input-data'
	type: 'Python'
	top: 'data'			# 待检测图像
	top: 'im_info'		# batch 数量、通道数
	top: 'gt_boxes'		# ground-truth 坐标
	python_param {
		module: 'roi_data_layer.layer'
		layer: 'RoIDataLayer'
		param_str: "'num_classes': 21"		# 类别数（背景 + 20 类目标）
	}
}
```

### AnchorTarget 层

```
layer {
	name: 'rpn-data'
	type: 'Python'
	bottom: 'rpn_cls_score'			# 为了获取特征图维度
	bottom: 'gt_boxes'				# ground-truth
	bottom: 'im_info'				# 图像 shape 信息
	bottom: 'data'					# 待预测图像，代码中似乎未使用
	top: 'rpn_labels'					# Anchor 对应标签
	top: 'rpn_bbox_targets'				# BBox 目标
	top: 'rpn_bbox_inside_weights'		# 用于计算 RPN bbox loss
	top: 'rpn_bbox_outside_weights'		# 用于计算 RPN bbox loss
	python_param {
		module: 'rpn.anchor_target_layer'
		layer: 'AnchorTargetLayer'
		param_str: "'feat_stride': 16"
	}
}
```

- AnchorTarget 层处理过程：

	- 根据 Anchor 得到原始图像的对应区域

	- 删除超出图像边界的区域对应的 Anchor

	- 计算 Anchor 对应区域和 Ground Truth 的 \\(IoU\\)

	- 标记正样本、负样本、无效样本

		- 正样本：

			- 将最大 \\(IoU\\) 对应的 Anchor 作为正样本

			- Anchor 与任一 Ground Truth 的 \\(IoU \geq\\) ```RPN_POSITIVE_OVERLAP```

		- 负样本：

			- Anchor 与所有 Ground Truth 的 \\(IoU \geq\\) ```RPN_NEGATIVE_OVERLAP```

		- 无效样本：

			- 既不属于正样本，也不属于负样本的其他样本

	- 正、负样本太多时，随机采样

	- 根据 Anchor 的对应区域，计算 BBox 的目标

### ProposalTarget 层

```
layer {
	name: 'proposal'
	type: 'Python'
	bottom: 'rpn_cls_prob_reshape'		# RPN softmax 输出
	bottom: 'rpn_bbox_pred'				# RPN bbox 输出
	bottom: 'im_info'					# 图像 shape
	top: 'rpn_rois'						# Region Proposal
	python_param {
		module: 'rpn.proposal_layer'
		layer: 'ProposalLayer'
		param_str: "'feat_stride': 16"
	}
}
layer {
	name: 'roi-data'			# ProposalTarget 层是为了方便计算 loss
	type: 'Python'
	bottom: 'rpn_rois'			# Region Proposal
	bottom: 'gt_boxes'			# Ground Truth
	top: 'rois'						# 选取部分 Region Proposal
	top: 'labels'					# Region Proposal label
	top: 'bbox_targets'				# Region Proposal bbox 目标
	top: 'bbox_inside_weights'		# 用于计算 Fast R-CNN bbox loss
	top: 'bbox_outside_weights'		# 用于计算 Fast R-CNN bbox loss
		python_param {
		module: 'rpn.proposal_target_layer'
		layer: 'ProposalTargetLayer'
		param_str: "'num_classes': 21"
	}
}
```

### Loss 层

```
layer {
	name: "rpn_loss_cls"
	type: "SoftmaxWithLoss"				# RPN softmax loss
	bottom: "rpn_cls_score_reshape"
	bottom: "rpn_labels"
	propagate_down: 1
	propagate_down: 0
	top: "rpn_cls_loss"
	loss_weight: 1						# loss 权重
	loss_param {
		ignore_label: -1
		normalize: true
	}
}
layer {
	name: "rpn_loss_bbox"
	type: "SmoothL1Loss"				# RPN bbox loss
	bottom: "rpn_bbox_pred"
	bottom: "rpn_bbox_targets"
	bottom: 'rpn_bbox_inside_weights'
	bottom: 'rpn_bbox_outside_weights'
	top: "rpn_loss_bbox"
	loss_weight: 1						# loss 权重
	smooth_l1_loss_param {
		sigma: 3.0
	}
}
layer {
	name: "loss_cls"
	type: "SoftmaxWithLoss"				# Fast R-CNN softmax loss
	bottom: "cls_score"
	bottom: "labels"
	propagate_down: 1
	propagate_down: 0
	top: "loss_cls"
	loss_weight: 1						# loss 权重
}
layer {
	name: "loss_bbox"
	type: "SmoothL1Loss"				# Fast R-CNN bbox loss
	bottom: "bbox_pred"
	bottom: "bbox_targets"
	bottom: "bbox_inside_weights"
	bottom: "bbox_outside_weights"
	top: "loss_bbox"						# loss 权重
	loss_weight: 1
}
```

## 网络输出

- RPN 网络 Softmax 输出 ```rpn_cls_prob_reshape``` 维度为 \\((1, \ 18, \ H, \ W)\\)

	- \\(18\\) 表示每一种 Anchor 是目标、背景的概率

	- \\(H, \ W\\) 是特征图维度

- RPN 网络 BBox 输出 ```rpn_bbox_pred``` 维度为 \\((1, \ 36, \ H, \ W)\\)

	- \\(36\\) 表示每一种 Anchor 对应的 \\(4\\) 个 BBox 变换量

	- \\(H, \ W\\) 是特征图维度

- RPN 网络的最终输出 ```rois``` 维度为 \\((N, \ 5)\\)

	- \\(N\\) 是 Region Proposal 数量

	- \\(5\\) 维向量分别是 \\(id, \ x1, \ y1, \ x2, \ y2\\)

		- \\(id\\) 表示当前图像在 batch 内的序号

		- 由于只支持对单张图片检测，\\(id = 0\\)

- Fast R-CNN Softmax 模块的输出 ```cls_prob``` 维度为 \\((N, \ C)\\)

	- \\(N\\) 是 Region Proposal 数量

	- \\(C\\) 是每一个 Region Proposal 属于每一类的概率

- Fast R-CNN BBox 模块的输出 ```bbox_pred``` 维度为 \\((N, \ 4C)\\)

	- \\(N\\) 是 Region Proposal 数量

	- \\(4C\\) 表示每一类对应的 \\(4\\) 个 BBox 变换量

- 先根据 ```rois``` 和 ```bbox_pred``` 进行 BBox 回归，再按 ```cls_prob ``` 进行 NMS