import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
import torch
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/Qwen3-8B-SFT-unsloth"
model_name = "Qwen/Qwen3-8B" 

# 使用unsloth加载模型
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    # use_exact_model_name=True,
    # local_files_only=True,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True # 加载4bit模型
)

# 加载LoRA适配器
model = FastLanguageModel.get_peft_model(
    model=model,
    r=8,
    target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
)
dataset_dict = load_dataset('json', data_files={"train": "data/keywords_data_train.jsonl",
                                                "test": "data/keywords_data_test.jsonl"})

# 将数据转为标准对话格式（OpenAI）
def map_func(example):
    conversation = example['conversation']
    messages = []
    for item in conversation:
        messages.append({'role': 'user', 'content': item['human']})
        messages.append({'role': 'assistant', 'content': item['assistant']})
    return {'messages': messages}

dataset_dict = dataset_dict.map(map_func, batched=False,
                                remove_columns=['dataset', 'conversation', 'category', 'conversation_id'])

# 将对话格式的数据转为字符串（Chat Templete）
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3",  # 使用Qwen3的对话模板
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {'text': texts}

dataset_dict = dataset_dict.map(formatting_prompts_func, batched=True, remove_columns=['messages'])

# Configure trainer
training_args = SFTConfig(
    output_dir="./finetuned/Qwen3-8B-SFT-unsloth",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=3,
    learning_rate=5e-5,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,
    warmup_steps=0.1,
    report_to=["tensorboard"]
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    # 不需要再传入LoraConfig，因为model已经定义好了和LoRA适配器相关的内容
    # peft_config=lora_config
)

trainer.train()
# 保存LoRA适配器
trainer.save_model("./finetuned/Qwen3-8B-SFT-unsloth")

# 保存合并模型
model.save_pretrained_merged("./finetuned/Qwen3-8B-SFT-unsloth-merged", tokenizer, save_method="merged_16bit")
