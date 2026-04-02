# SFTConfig 参数整理

`SFTConfig` 来自 `trl`，本质上可以理解成：

- 在 `transformers.TrainingArguments` 的基础上
- 增加了一些 `SFT`（监督微调）专用参数



---

## 1. 先用一句话理解整份配置

一套 `SFTConfig`，主要就是在回答这几个问题：

- 训练要跑多久
- 每次喂多少数据
- 用什么优化器和学习率策略
- 怎么省显存、提速度
- 多久记录日志、评估、保存
- 数据怎么截断、打包、算 loss


---

## 2. 训练规模与训练时长

这组参数决定“训练跑多大、跑多久”。

### `output_dir`

- 输出目录
- 用来保存模型、checkpoint、日志等内容

### `per_device_train_batch_size`

- 每张卡每一步喂多少条训练样本
- 如果是单卡，就是单步 batch size
- 如果是多卡，总 batch 还要乘设备数

### `num_train_epochs`

- 训练多少轮
- 一轮表示把整个训练集完整跑一遍

### `max_steps`

- 最大训练步数
- 如果设置为正数，通常会优先生效
- 常见理解：`num_train_epochs` 控制“按轮数训练”，`max_steps` 控制“按步数训练”

### `gradient_accumulation_steps`

- 梯度累积步数
- 显存不够时非常常用
- 比如每步只跑 1 条，但累积 8 步再更新一次，相当于有效 batch 变大了

### `max_grad_norm`

- 梯度裁剪阈值
- 防止梯度爆炸

### `seed`

- 训练随机种子
- 控制初始化、shuffle 等随机行为

### `resume_from_checkpoint`

- 从指定 checkpoint 恢复训练

### `ignore_data_skip`

- 恢复训练时，是否忽略已经处理到的数据位置
- 一般不需要改

### 有效 batch size 的理解

常见公式：

`有效 batch size = per_device_train_batch_size × 设备数 × gradient_accumulation_steps`

例如：

- 单卡
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`

那么有效 batch size 可以理解为 `8`

---

## 3. 学习率与优化器

这组参数决定“模型参数怎么更新”。

### `learning_rate`

- 初始学习率
- 是最核心的超参数之一

### `lr_scheduler_type`

- 学习率调度器类型
- 常见如：
  - `linear`
  - `cosine`
  - `constant`

### `lr_scheduler_kwargs`

- 调度器的额外配置

### `warmup_steps`

- 预热步数
- 训练刚开始时让学习率从较小值逐渐升上去

### `warmup_ratio`

- 预热比例
- 和 `warmup_steps` 通常二选一

### `optim`

- 优化器类型
- 比如 `adamw_torch_fused`

### `optim_args`

- 传给优化器的额外参数

### `weight_decay`

- 权重衰减
- 用来抑制过拟合，属于正则化手段

### `adam_beta1` / `adam_beta2` / `adam_epsilon`

- AdamW 这类优化器内部使用的超参数
- 一般保持默认即可


### 这组参数怎么记

可以压缩成四句话：

- `learning_rate`：步子迈多大
- `scheduler`：训练过程中步子怎么变化
- `optim`：用什么规则更新参数
- `weight_decay`：防止模型学得太“野”

---

## 4. 精度、显存与性能优化

这一组在实际训练里非常重要，尤其是单卡训练。

### `bf16`

- 是否启用 `bfloat16`
- 如果显卡支持，通常优先于 `fp16`

### `fp16`

- 是否启用 `float16`
- 常见于混合精度训练

### `bf16_full_eval` / `fp16_full_eval`

- 评估时是否也使用对应半精度

### `tf32`

- 是否允许 TensorFloat-32
- 常用于支持 TF32 的 NVIDIA GPU 上加速矩阵运算

### `gradient_checkpointing`

- 梯度检查点
- 通过“多算一点”来“少占一些显存”
- 大模型训练里很常见

### `gradient_checkpointing_kwargs`

- 梯度检查点的额外配置

### `use_liger_kernel`

- 是否启用 liger kernel
- 某些场景下可以提升速度或降低显存占用

### `liger_kernel_config`

- liger kernel 的详细配置


### `torch_empty_cache_steps`

- 每隔多少步主动清一次 CUDA cache

### `auto_find_batch_size`

- 自动寻找不会 OOM 的 batch size

### `activation_offloading`

- 将部分激活转移到别的位置（例如 CPU）
- 用时间换显存

### 单卡训练最值得先关注的几个

- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `bf16` 或 `fp16`
- `gradient_checkpointing`
- `max_length`

---

## 5. 日志相关

这组参数决定“训练过程怎么观察”。

### `logging_strategy`

- 什么时候记录日志
- 常见是 `steps`

### `logging_steps`

- 每多少步记录一次日志

### `logging_first_step`

- 第一步是否就记录日志

### `log_on_each_node`

- 多机训练时，每个节点是否都记录日志

### `logging_nan_inf_filter`

- 是否过滤异常的 `nan/inf` 日志

### `include_num_input_tokens_seen`

- 是否记录累计看过多少 token

### `log_level` / `log_level_replica`

- 日志等级

### `disable_tqdm`

- 是否关闭进度条

### `report_to`

- 日志汇报到哪里
- 常见有：
  - `none`
  - `tensorboard`
  - `wandb`

### `run_name`

- 当前训练任务的名称

### `project`

- 项目名

### `trackio_space_id`

- 与 trackio 相关的配置
- 如果你不用对应平台，通常不用关心

### `logging_dir`

- 日志目录

---

## 6. 评估相关

这组参数决定“什么时候验证模型效果”。

### `eval_strategy`

- 是否评估，以及按什么节奏评估
- 常见：
  - `no`
  - `steps`
  - `epoch`

### `eval_steps`

- 每多少步评估一次

### `eval_delay`

- 延迟到多少步之后才开始评估

### `per_device_eval_batch_size`

- 每张卡的评估 batch size

### `prediction_loss_only`

- 评估时是否只返回 loss

### `eval_on_start`

- 训练开始前先评估一次

### `eval_do_concat_batches`

- 是否将评估 batch 拼接后再统一处理

### `eval_use_gather_object`

- 分布式评估时是否 gather Python object

### `eval_accumulation_steps`

- 评估阶段的累积步数
- 有时可以帮助控制评估显存

### `include_for_metrics`

- 哪些额外内容需要传入 metric 计算

### `batch_eval_metrics`

- 是否以 batch 级别计算 metric

---

## 7. 保存与最佳模型选择

这组参数决定“多久存一次、保留多少、是否加载最佳模型”。

### `save_strategy`

- 保存策略
- 常见：
  - `steps`
  - `epoch`

### `save_steps`

- 每多少步保存一次

### `save_on_each_node`

- 多机时每个节点都保存吗

### `save_total_limit`

- 最多保留多少个 checkpoint
- 超过后会删除更早的 checkpoint

### `save_only_model`

- 是否只保存模型权重
- 如果为 `True`，优化器等训练状态可能不会保存

### `enable_jit_checkpoint`

- 与 JIT/checkpoint 相关的高级选项

### `load_best_model_at_end`

- 训练结束后是否自动加载表现最好的 checkpoint

### `metric_for_best_model`

- 用哪个指标判断“最佳模型”

### `greater_is_better`

- 指标越大越好还是越小越好
- 比如 accuracy 通常越大越好，loss 通常越小越好

### `restore_callback_states_from_checkpoint`

- 恢复训练时是否一起恢复 callback 状态

---




## 10. DataLoader 与数据读取

这组参数决定“数据怎么从磁盘送到模型里”。

### `dataloader_drop_last`

- 最后一个不足 batch 的小批次是否丢弃

### `dataloader_num_workers`

- 数据加载 worker 数量

### `dataloader_pin_memory`

- 是否启用 pin memory
- GPU 训练一般保留默认即可

### `dataloader_persistent_workers`

- worker 是否常驻

### `dataloader_prefetch_factor`

- 每个 worker 预取多少批数据

### `remove_unused_columns`

- 是否自动删除模型用不到的数据列

### `label_names`

- 指定哪些列是 label

### `train_sampling_strategy`

- 训练采样策略

### `length_column_name`

- 表示样本长度的列名

---

## 11. 训练流程开关

### `do_train`

- 是否执行训练

### `do_eval`

- 是否执行评估

### `do_predict`

- 是否执行预测

### `skip_memory_metrics`

- 是否跳过显存统计

---

## 12. SFT 专属参数

这一组是 `SFTConfig` 相比普通 `TrainingArguments` 更值得重点关注的部分。

### `model_init_kwargs`

- 模型初始化时传入的额外参数

### `chat_template_path`

- 聊天模板路径
- 用于把多轮对话格式化成模型输入文本

### `dataset_text_field`

- 数据集中哪一列存放文本
- 默认通常是 `text`

### `dataset_kwargs`

- 数据集额外参数

### `dataset_num_proc`

- 数据预处理并行进程数

### `eos_token`

- 结束 token

### `pad_token`

- padding token

### `max_length`

- 每条样本允许的最大 token 长度
- 超过就需要截断

### `truncation_mode`

- 截断策略
- 比如 `keep_start` 表示保留前面部分

### `shuffle_dataset`

- 是否打乱数据集

### `packing`

- 是否把多条短样本拼接成一条更长的序列
- 当样本很短时，这个参数常常能提高 token 利用率

### `packing_strategy`

- 拼接策略
- 例如 `bfd`

### `padding_free`

- 尽量减少 padding 的处理方式

### `pad_to_multiple_of`

- 将长度 pad 到指定倍数
- 有时对硬件效率更友好

### `eval_packing`

- 评估时是否也使用 packing

### `completion_only_loss`

- 是否只对 completion 部分计算 loss
- 适用于 prompt-completion 格式的数据

### `assistant_only_loss`

- 是否只对 assistant 回复部分计算 loss
- 很适合多轮对话 SFT

### `loss_type`

- 损失函数类型
- 默认一般是 `nll`

### `activation_offloading`

- 激活是否卸载到其他设备或内存
- 本质是“用时间换显存”

---

## 13. 最值得优先理解的一小撮参数

如果你是第一次做单卡 SFT，不要试图一次记住全部参数。先抓下面这些最关键：

- `output_dir`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs` 或 `max_steps`
- `learning_rate`
- `lr_scheduler_type`
- `warmup_steps` 或 `warmup_ratio`
- `bf16` 或 `fp16`
- `gradient_checkpointing`
- `logging_steps`
- `save_steps`
- `eval_strategy`
- `dataset_text_field`
- `max_length`
- `packing`
- `assistant_only_loss` / `completion_only_loss`

如果这几个已经理解得比较清楚，基本就能开始配置一次正常的 SFT 训练了。

---

## 14. 一份常见的单卡 SFT 起步配置

下面这份配置可以作为理解参数时的参考模板：

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,  # 如果显卡支持；否则可考虑 fp16=True
    gradient_checkpointing=True,
    max_length=1024,
    packing=False,
    report_to="none",
)
```

这份配置背后的意思其实很简单：

- 单步 batch 很小，靠梯度累积扩大有效 batch
- 开启混合精度和梯度检查点来省显存
- 保持比较标准的学习率和训练轮数
- 日志、保存频率都设置成比较容易观察的水平

---

## 15. 一个适合记忆的总框架

以后再看到这长串参数，可以直接按下面这个顺序去想：

### 第一步：训练怎么跑

- `batch size`
- `epoch / steps`
- `gradient accumulation`

### 第二步：参数怎么更新

- `learning rate`
- `optimizer`
- `scheduler`

### 第三步：显存和速度怎么平衡

- `bf16/fp16`
- `gradient checkpointing`
- `max_length`

### 第四步：过程怎么管理

- `logging`
- `eval`
- `save`

### 第五步：数据怎么喂给模型

- `dataset_text_field`
- `chat_template_path`
- `packing`
- `truncation`
- `assistant_only_loss` / `completion_only_loss`

如果按这个框架理解，`SFTConfig` 就不会显得是一堆杂乱无章的参数表，而是一整套“训练控制面板”。
