# 1、构建一个BitsAndBytesConfig对象
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 使用4bit量化加载
    bnb_4bit_quant_type="nf4", # 使用nf4量化类型
    bnb_4bit_use_double_quant=False, # 基于实际环境来决定是否需要开启双重量化，开启后，会让反量化过程得到的数值精度更低，但是能够让占用的显存更少
)

# 2、加载模型时传入BitsAndBytesConfig对象
from transformers import AutoModelForCausalLM,AutoTokenizer
quantized_model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B",quantization_config=bnb_config)
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B")

from peft import prepare_model_for_kbit_training
quantized_model = prepare_model_for_kbit_training(quantized_model)

# 1、引入Peft
from peft import LoraConfig
# 2、构建一个config对象
peft_config = LoraConfig(
    r  = 8,
    lora_alpha= 8,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# 1、引入sft trainer
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/Qwen3-8B-QLoRA"
# 2、加载数据集
from datasets import load_dataset
dataset = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})
# 3、定义一个map函数，将数据集中的每个样本，转换成SFTTrainer支持的格式
def map_to_sft_format(example):
    conversation = example["conversation"]
    message_list = []
    for conv in conversation:
        for key ,value in conv.items():
            key = "user" if key == "human" else key
            message_list.append({"role":key,"content":value})

    return {"messages":message_list}
# 4、对dataset 进行转换
remove_list = list(dataset["train"][0].keys())
dataset = dataset.map(map_to_sft_format,batched=False,remove_columns=remove_list)
# 5、构造SFTConfig对象
config = SFTConfig(
    output_dir = "finetuned/Qwen3-8B-QLoRA",
    per_device_train_batch_size = 3,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-5,
    # max_steps = 3000,
    # 日志
    logging_steps = 100,
    num_train_epochs=1,
    
    report_to = ["tensorboard"],
    # 显存优化相关:
    bf16=True, # 混合精度
    gradient_checkpointing=True, # 梯度检查点
    activation_offloading = False, # CPU 卸载2
    # 保存相关
    save_strategy = "steps",
    save_steps = 300,
    # 评估相关
    eval_steps = 300,
    eval_strategy = "steps",
    metric_for_best_model = "eval_loss",
    load_best_model_at_end=True,
    greater_is_better = False,
    max_length = 2500,
    
    chat_template_path="chat_template.jinja",
    assistant_only_loss=True
)

# 6、构造SFTTrainer对象
from transformers import AutoModelForCausalLM,AutoTokenizer
for name,module in model.named_modules():
    print(name,module)
trainer = SFTTrainer(
    model = quantized_model,
    processing_class = tokenizer,
    args = config,
    train_dataset= dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config
)

trainer.train()
# 此时保存的就是AB矩阵，而不是原模型的所有的参数
trainer.save_model("finetuned/Qwen3-8B-QLoRA")