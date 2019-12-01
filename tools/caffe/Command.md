# 基本命令

## 基本命令

```shell
caffe <cmd> <args>
```

### $\mathrm{cmd}$ 选项

- `train`：训练或 $\mathrm{fine \ tuning}$

- `test`：测试

- `device_query`：显示 $\mathrm{GPU}$ 信息

- `time`：显示程序执行时间

### $\mathrm{args}$ 选项

- `--solver`：必选参数，超参数配置文件

- `--gpu`：使用的 $\mathrm{GPU}$：

  - 多块 $\mathrm{GPU}$ 用 `,` 隔开

  - `--gpu all` 表示使用所有 $\mathrm{GPU}$

- `--weights`：$\mathrm{fine \ tuning}$ 时的预训练模型

- `--snapshot`：从快照中恢复训练，不能与 `--weights` 同时使用

- `--iterations`：迭代次数，默认为 $50$