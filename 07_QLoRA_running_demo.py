from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from peft import LoraConfig
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import TrainerCallback
import torch
from transformers import TrainerCallback
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/Qwen3-8B-SFT-QLoRA"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = "model/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config) 
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
torch.cuda.empty_cache()
dataset_dict = load_dataset('json',data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})

def map_func(example):
    conversation = example["conversation"]

    messages = []

    for item in conversation:
        messages.append({"role":"user","content":item["human"]})
        messages.append({"role":"assistant","content":item["assistant"]})
   
    return {"messages":messages}

dataset_dict = dataset_dict.map(function=map_func,batched=False,remove_columns=["dataset","conversation","category","conversation_id"])

peft_config = LoraConfig(
    r = 8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig( 
    output_dir="./finetuned/Qwen3-8B-SFT-QLoRA",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=3,
    num_train_epochs=1,
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

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.train()
trainer.save_model("./finetuned/Qwen3-8B-SFT-QLoRA")
