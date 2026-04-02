from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import AutoModelForCausalLM
from peft import LoraConfig

lora_config = LoraConfig(
    r = 4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules=[""]


)
