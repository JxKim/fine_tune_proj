# LoraConfig 参数解释

`LoraConfig` 是 PEFT 中用于配置 `LoRA` 适配器的核心配置类，主要决定：

- LoRA 要挂载到哪些模块
- LoRA 的秩、缩放、dropout 等训练行为
- 是否启用一些 LoRA 的扩展变体，如 `RSLoRA`、`DoRA`、`aLoRA`
- 是否使用特殊初始化方式，如 `PiSSA`、`EVA`、`OLoRA`、`CorDA`、`LoftQ`

这份文档基于 `peft.tuners.lora.config.LoraConfig` 的源码注释整理，适合作为训练时的参数速查说明。

## 1. 基础参数

### `r: int`

LoRA 的秩，也就是低秩分解中的维度大小。

- 值越小，可训练参数越少，显存占用越低
- 值越大，表达能力通常越强，但训练成本也更高
- 常见取值有 `4`、`8`、`16`、`32`

### `lora_alpha: int`

LoRA 的缩放系数。

- 经典 LoRA 中，实际缩放一般与 `lora_alpha / r` 有关
- 如果启用 `use_rslora=True`，缩放会变成 `lora_alpha / sqrt(r)`

### `lora_dropout: float`

LoRA 层上的 dropout 概率。

- 用于训练阶段正则化
- 常见取值为 `0.0`、`0.05`、`0.1`

### `bias: str`

控制是否训练偏置项，可选值：

- `none`
- `all`
- `lora_only`

如果设置为 `all` 或 `lora_only`，即使禁用 adapter，模型输出也可能与原始基座模型不同。

### `lora_bias: bool`

是否为 LoRA 的 `B` 参数启用 bias，默认是 `False`。


### `task_type: Optional[Union[str, TaskType]]`

指定当前 PEFT 适配的任务类型。

这个参数虽然不一定会在 `LoraConfig` 这段注释里单独展开，但在实际使用时非常重要，因为它会影响：

- PEFT 如何识别当前模型任务
- 适配器如何包装模型
- 某些任务下前向逻辑和保存逻辑的处理方式

常见取值包括：

- `CAUSAL_LM`：因果语言模型，例如 Qwen、LLaMA、GPT 一类自回归生成模型
- `SEQ_2_SEQ_LM`：序列到序列生成任务，例如 T5、BART
- `SEQ_CLS`：序列分类任务
- `TOKEN_CLS`：Token 分类任务
- `QUESTION_ANS`：问答任务
- `FEATURE_EXTRACTION`：特征提取任务

在大模型 SFT 里，最常见的是：CAUSAL_LM



## 2. 目标模块相关参数

### `target_modules: Optional[Union[List[str], str]]`

指定要应用 LoRA 的模块名称。

- 传字符串：按正则匹配模块名
- 传字符串列表：按精确匹配，或匹配模块名后缀
- 传 `all-linear`：对所有线性层或 `Conv1D` 层应用 LoRA
- 传空列表 `[]`：通常表示配合 `target_parameters` 使用

### `exclude_modules: Optional[Union[List[str], str]]`

指定哪些模块不要应用 LoRA。

### `target_parameters: Optional[List[str]]`

直接对参数名应用 LoRA，而不是对模块应用 LoRA，适合 MoE 等直接使用 `nn.Parameter` 的结构。

示例：

```python
target_parameters = [
    "feed_forward.experts.gate_up_proj",
    "feed_forward.experts.down_proj",
]
```

### `modules_to_save: List[str]`

除了 LoRA 层之外，额外指定哪些模块也要参与训练并保存。


### `layers_to_transform: Union[List[int], int]`

指定只对哪些层编号应用 LoRA。

### `layers_pattern: Optional[Union[List[str], str]]`

配合 `layers_to_transform` 使用，用来指定层列表所在的模块名，例如 `layers` 或 `h`。

### `layer_replication: List[Tuple[int, int]]`

基于原始模型层构建新的层堆叠结构，新层会拥有各自独立的 LoRA adapter。


## 4. 模式化定制

### `rank_pattern: dict`

为不同层单独指定不同的 rank。

### `alpha_pattern: dict`

为不同层单独指定不同的 `lora_alpha`。

## 5. 初始化相关参数

### `init_lora_weights`

可选值包括：

- `True`
- `False`
- `"gaussian"`
- `"eva"`
- `"olora"`
- `"pissa"`
- `"pissa_niter_[iters]"`
- `"corda"`
- `"loftq"`
- `"orthogonal"`

说明：

- `True`：默认初始化，LoRA 初始通常是指A随机初始化，B初始化为0，一开始LoRA对原模型几乎没影响，no-op
- `False`：随机初始化，主要用于调试




## 6. 常见示例

### 标准 Causal LM LoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

### 只对部分层生效

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[20, 21, 22, 23],
    layers_pattern="layers",
    task_type="CAUSAL_LM",
)
```

### 使用 `target_parameters`

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[],
    target_parameters=[
        "feed_forward.experts.gate_up_proj",
        "feed_forward.experts.down_proj",
    ],
    task_type="CAUSAL_LM",
)
```

## 7. 实践建议

- 常用核心参数通常只有 `r`、`lora_alpha`、`lora_dropout`、`target_modules`、`bias`

- 不确定目标层时，先检查模型结构中的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`

## 8. 一句话总结

`LoraConfig` 本质上是在控制三件事：

- LoRA 打到哪里
- LoRA 怎么训练
- LoRA 怎么初始化和扩展
