"""
RLHF当中，从BaseModel训练成InstructModel
"""

import re
from typing import Dict, List,Optional
from transformers import AutoTokenizer,AutoModelForCausalLM,PreTrainedTokenizerFast,PretrainedBartModel
# from transformers import DataCollatorForLanguageModeling # 用于语言模型的训练数据collator
import torch
import numpy as np
from dataclasses import dataclass
import datasets
import time
from torch.utils.tensorboard import SummaryWriter
model = None
tokenizer:Optional[PreTrainedTokenizerFast] = None

# 1、SFT相关配置
@dataclass
class SFTConfig:
    max_length:int = 2500
    batch_size:int = 2
    log_iter:int = 250
    log_dir:str = "logs/Qwen3-0.6B-no-trl-sft"
    max_lr:float = 2e-5
    min_lr:float = 2e-6
    warmup_ratio:float = 0.1
    device:str = "cuda"
    train_data_size:int = 50000

# 2、加载模型，tokenizer，并设置生成配置
def load_model_tokenizer():
    """
    加载模型和tokenizer
    """
    global model,tokenizer
    if model is None or tokenizer is None:
        model_path = "./model/Qwen3-0.6B-Base"
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model,tokenizer
    return model,tokenizer


# 3、加载数据集，并对其进行tokenize处理
def get_data_ultrachat_200k(config:SFTConfig):
    # 2、下载并处理数据集
    # 2.1 加载UltraChat 200k数据集
    ultrachat_200k_data:datasets.DatasetDict = datasets.load_dataset("./data/ultrachat_200k")
    tokenizer = load_model_tokenizer()[1]
    train_data = []
    i = 0
    # 2.2 对数据进行tokenize处理
    while True:
        data:List = ultrachat_200k_data["train_sft"][i]["messages"]
        
        data.insert(0,{"role":"system","content":"You are a helpful assistant."})
        input_ids = tokenizer.apply_chat_template(data,tokenize=True,add_generation_prompt=False,truncation=True,max_length=2500)
        
        train_data.append(input_ids)
        i += 1
        if i % 1000 == 0:
            print(f"已经处理了 {i} 条数据")
        # 仅训练train_data_size条数据
        if i == config.train_data_size:
            break
    
    return train_data

# 2.3 构建数据集当中answer mask: 后续训练过程中，只关注answer部分的loss
def create_answer_mask(input_ids,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    Args:
        input_ids (_type_): _description_
        tokenizer (PreTrainedTokenizerFast): _description_
    Returns:
        _type_: _description_
    """

    
    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵
    answer_mask = torch.zeros_like(input_ids)


    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    for idx,ids in enumerate(input_ids):
        # 获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()

        # 排除第一个eos_position: 第一个对应的是system prompt
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[11,21]
        assistant_ends:[16,26]
    Args:
        eos_positions (List[int]): _description_
    Returns:
        _type_: _description_
    """

    use_ends = [pos+1 for pos in eos_positions[::2]]
    assistant_ends = [pos+1 for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[11,21]，assistant_ends:[16,26]

    user_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    assistant_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    要想获取到assistant的回答的起始位置，就需要再跳过\n,<|im_start|>,assistant 这三个token，所以需要加3.
    要想获取到assistant的回答的结束位置，就需要往前跳一个<|im_end|>，所以需要减1.
    Args:
        mask (_type_): _description_
        user_ends (_type_): _description_
        assistant_ends (_type_): _description_
        seq_len (_type_): _description_
    Returns:
        _type_: _description_
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1

def cosine_decay(current_step,warmup_ratio,total_steps,max_lr,min_lr):
    warmup_steps = int(warmup_ratio * total_steps)
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (max_lr - min_lr) * decay

def compute_loss(output_logits,target_labels,assistant_answer_mask):
    """
    对于单个step，计算前向传播后的损失：
    """
    # 对output_logits进行softmax处理，得到每个token的概率
    log_probabilities = torch.log(torch.softmax(output_logits,dim=-1))
    log_probabilities = torch.log_softmax(output_logits,dim=-1)
    # 通过gather函数，得到model输出的target token处所对应的位置
    gathered_log_probs = torch.gather(
        log_probabilities,
        dim=-1,
        index=target_labels.unsqueeze(-1)
    )

    # 计算负对数似然损失
    negative_log_likelihood = gathered_log_probs*(-1)

    token_losses = negative_log_likelihood.squeeze(-1)

    # 对token_losses进行mask，只关注answer部分的loss
    masked_token_losses = torch.mul(token_losses,assistant_answer_mask)

    # 只对batch中所有有效token做平均，更接近SFTTrainer默认的token-level mean口径
    valid_token_count = assistant_answer_mask.sum()
    average_loss = masked_token_losses.sum() / valid_token_count

    return average_loss

# 3.3 模型训练过程
def train(model,config,tokenizer):
    device = config.device
    train_data = get_data_ultrachat_200k(config=config)
    model.train()
    model.to(device)
    total_steps  = len(train_data) // config.batch_size
    print(f"总训练步数为{total_steps}")
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.max_lr)
    writer = SummaryWriter(log_dir=config.log_dir)
    step_losses = []
    skipped_batches_count = 0
    import tqdm
    progress_bar = tqdm.tqdm(total=total_steps, desc="step")
    for step in range(total_steps):
        # 1.获取数据
        batch = train_data[step*config.batch_size:(step+1)*config.batch_size]
        
        # 使用 <|endoftext|> 作为pad token
        pad_token_id:int = tokenizer.pad_token_id
        max_len = max(len(ids['input_ids']) for ids in batch)
        
        padded_ids = []
        
        # 对batch进行padding，使整个batch的序列长度相同
        for ids in batch:
            # print('当前的ids为',ids)
            input_ids = ids["input_ids"]
            padding_length = max_len - len(input_ids)
            padded_sequence = torch.nn.functional.pad(torch.tensor(input_ids,dtype=torch.long), (0, padding_length), value=pad_token_id).tolist()
            padded_ids.append(padded_sequence) 
            
        

        # 2. 构建input_ids和output_ids，（如果不手写实现，可以使用transformers的DataCollatorForLanguageModeling）
        batch_input_tensor = torch.tensor(padded_ids)
        model_inputs = batch_input_tensor[:,:-1].to(device)
        target_labels = batch_input_tensor[:,1:].to(device)

        padding_mask = torch.where(target_labels == pad_token_id,0,1)
        assistant_answer_mask = create_answer_mask(model_inputs,tokenizer).to(device)
        

        final_loss_mask = assistant_answer_mask & padding_mask

        # 判断当前数据当中是否存在无有效的answer token的情况，如果有则跳过
        tokens_per_sample = final_loss_mask.sum(dim=-1)
        min_answer_tokens = tokens_per_sample.min().item()
        if min_answer_tokens == 0:
            print(f"当前step {step+1}中，无有效answer token，跳过该批次")
            skipped_batches_count+=1
            progress_bar.update(1)
            continue

        # 3. 执行前向传播
        model_logits = model(model_inputs).logits

        # 4、计算当前step的平均loss
        step_loss = compute_loss(output_logits=model_logits,target_labels=target_labels,assistant_answer_mask=final_loss_mask)


        # 5、反向传播并更新参数
        step_loss.backward()
        current_learning_rate = cosine_decay(
            current_step=step+1,
            warmup_ratio=config.warmup_ratio,
            total_steps=total_steps,
            max_lr=config.max_lr,
            min_lr=config.min_lr
        )

        writer.add_scalar("train_lr",current_learning_rate,step+1)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate
        optimizer.step()
        optimizer.zero_grad()
        step_losses.append(step_loss.item())
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{step_losses[-1]:.4f}", lr=f"{current_learning_rate:.2e}")

        should_log = (step + 1) % config.log_iter == 0 or (step + 1) == total_steps
        if should_log:
            recent_losses = step_losses[-config.log_iter:]
            recent_average_loss = np.nanmean(recent_losses)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(
                f"时间：{current_time} | "
                f"步数：{step+1}/{total_steps} | "
                f"最近{len(recent_losses)}批次平均损失：{recent_average_loss:.4f} | "
                f"当前学习率：{current_learning_rate:.2e}"
            )
            writer.add_scalar("train_loss",recent_average_loss,step+1)

    print("训练完成")

def save_model_tokenizer(model,tokenizer,save_path):
    import os
    os.makedirs(save_path,exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"模型和分词器已保存至{save_path}")


def test_answer_mask():
    """
    测试掩码机制
    """
    message_list = [
        {"role":"system","content":"你是一个专业的翻译助手"},
        {"role":"user","content":"你好，你是谁"},
        {"role":"assistant","content":"我是一个翻译助手，我能为你做什么？"},
    ]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")
    res = tokenizer.apply_chat_template(message_list,add_generation_prompt=False,tokenize=False)
    print("------1、使用chat template之后的结果为")
    print(repr(res))
    print("-----------------\n\n")

    res_token_ids = tokenizer.apply_chat_template(message_list,add_generation_prompt=False,tokenize=True)
    print(res_token_ids)
    print("------2、使用chat template之后的token_ids为")
    for token in res_token_ids["input_ids"]:
        print('token_id为',token,"；token为",repr(tokenizer.decode(token)))
    print("-----------------\n\n")
    print("------3、使用create_answer_mask之后的token_ids为")
    token_id_tensor = torch.tensor([res_token_ids["input_ids"][:-1]],device="cuda",dtype=torch.long)
    assistant_answer_mask = create_answer_mask(token_id_tensor,tokenizer)
    assistant_answer_mask_list = assistant_answer_mask.tolist()[0]
    for token_id,mask in zip(res_token_ids["input_ids"][:-1],assistant_answer_mask_list):
        print('token_id为',token_id,"；token为",repr(tokenizer.decode(token_id)),"；mask为",mask)
def main():
    # 配置启动参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_size",type=int,default=10,help="训练数据集的大小")
    parser.add_argument("--batch_size",type=int,default=2,help="批次大小")
    parser.add_argument("--output_dir",type=str,default="./finetuned/Qwen3-0.6B-SFT",help="SFT模型输出目录")
    # 解析启动参数
    args = parser.parse_args()
    # 1、创建配置实例
    config = SFTConfig(train_data_size=args.train_data_size,batch_size=args.batch_size)
    # 2、获取model和tokenizer
    model,tokenizer = load_model_tokenizer()
    # 3、训练模型
    train(model=model,config=config,tokenizer=tokenizer)

    save_model_tokenizer(model,tokenizer,save_path=args.output_dir)
if __name__ == "__main__":
    main()
    