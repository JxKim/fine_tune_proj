from dataclasses import dataclass
import time
from typing import List,Dict
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,PreTrainedTokenizerFast
from torch.utils.tensorboard import SummaryWriter

# 1.DPOTrainConfig
@dataclass
class DPOTrainConfig:
    train_data_size:int = 100 # 整体训练数据量
    max_length:int = 1700 # 每个样本最大长度
    batch_size:int = 2 # 每个批次样本数量
    logging_iter:int = 50 # 每个日志迭代次数
    log_dir : str = "./logs/Qwen3-0.6B-dpo-full-2341"
    warmup_ratio:float = 0.1 # 预热比例
    max_lr:float = 2e-5 # 最大学习率
    min_lr:float = 2e-6 # 最小学习率
    device:str = "cuda" # 设备
    # DPO参数
    beta:float = 0.5 

# 2.加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("finetuned/Qwen3-0.6B-SFT-2315")
tokenizer:PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("finetuned/Qwen3-0.6B-SFT-2315")
ref_model = AutoModelForCausalLM.from_pretrained("finetuned/Qwen3-0.6B-SFT-2315")

# 3.加载数据集
def get_data_ultrafeedback_binarized(config:DPOTrainConfig):
    """
    1、加载数据集
    2、插入system prompt
    3、使用tokenizer进行tokenize
    """

    from datasets import load_dataset
    binarized_data:Dict[str,List[Dict]] = load_dataset("data/ultrafeedback_binarized")
    chosen_input_ids = []
    rejected_input_ids = []
    i = 0
    while True:
        # 插入system prompt是为了符合Qwen2.5的规范要求
        data = [{"role":"system","content":"You are a helpful assistant"}] + list(binarized_data["train_sft"][i]["chosen"])
        input_ids = tokenizer.apply_chat_template(data,tokenize=True,max_length=config.max_length,truncation=True,add_generation_prompt=False)
        
        chosen_input_ids.append(input_ids)
        
        
        rejected_data = [{"role":"system","content":"You are a helpful assistant"}] + list(binarized_data["train_sft"][i]["rejected"])
        input_ids = tokenizer.apply_chat_template(rejected_data,tokenize=True,max_length=config.max_length,truncation=True,add_generation_prompt=False)
        rejected_input_ids.append(input_ids)

        i += 1
        if i % 1000 == 0:
            print(f'已处理{i}条数据')
        if i ==config.train_data_size or i ==len(binarized_data["train_sft"]):
            print(f"偏好已处理{i}条数据，处理完毕")
            break
    
    return chosen_input_ids,rejected_input_ids
    

def _compute_average_log_probability(logits,target_labels,mask:torch.Tensor):
    """
    计算logits在target_labels上的平均log概率
    Args:
        logits (_type_): _description_
        target_labels (_type_): _description_
        mask (_type_): _description_
    Returns:
        _type_: _description_
    """

    log_probalitites = torch.log_softmax(logits,dim=-1)

    # 使用gather函数获取target_labels上的log概率
    gathered_log_probs = torch.gather(
        log_probalitites,
        dim=-1,
        index = target_labels.unsqueeze(2)
    ).squeeze(2)

    # 对gathered_log_probs进行mask操作，只计算mask=1部分的log概率，
    masked_log_probs = torch.mul(gathered_log_probs,mask)
    # 计算每个样本的平均log概率
    average_log_prob = masked_log_probs.sum(dim=-1) / mask.sum(dim=-1)

    return average_log_prob

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

def compute_loss(preferred_logits,rejected_logits,reference_preferred_logits,reference_rejected_logits,preferred_labels,rejected_labels,preferred_answer_mask,rejected_answer_mask,beta:float):
    """
    DPO损失函数计算：
        计算单个批次当中，模型生成偏好回答的平均token概率，和模型生成拒绝回答的平均token概率，以及参考模型生成偏好回答的平均token概率，参考模型生成拒绝回答的平均token概率
    Args:
        output_logits: 模型输出的logits
        ref_output_logits: 参考模型输出的logits
        preferred_labels: 偏好回答labels
        rejected_labels: 拒绝回答labels
        assistant_answer_mask: assistant回答的掩码
    """
    preferred_log_prob = _compute_average_log_probability(preferred_logits,preferred_labels,mask=preferred_answer_mask)

    rejected_log_prob = _compute_average_log_probability(rejected_logits,rejected_labels,mask=rejected_answer_mask)

    reference_preferred_log_prob = _compute_average_log_probability(reference_preferred_logits,preferred_labels,mask=preferred_answer_mask)
    reference_rejected_log_prob = _compute_average_log_probability(reference_rejected_logits,target_labels=rejected_labels,mask=rejected_answer_mask)

    final = (preferred_log_prob - rejected_log_prob) - (reference_preferred_log_prob - reference_rejected_log_prob)

    loss = -torch.nn.functional.logsigmoid(final * beta)

    step_average_loss = loss.mean()

    return step_average_loss

def train(model,ref_model,tokenizer,config:DPOTrainConfig):
    """
    主训练函数
    Args:
        tokenizer:
        config:

    Returns:
    """
    # 1、将模型和参考模型移动到指定设备，注意：参考模型需要设置为评估模式，不需要Dropout
    model.to(config.device)
    ref_model.to(config.device)
    model.train()
    ref_model.eval()
    training_losses = []

    chosen_input_ids,rejected_input_ids = get_data_ultrafeedback_binarized(config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.max_lr)
    writer = SummaryWriter(config.log_dir)
    model.zero_grad()
    skipped_batches_count = 0
    import tqdm
    
    total_steps = len(chosen_input_ids) // config.batch_size if len(chosen_input_ids) % config.batch_size == 0 else len(chosen_input_ids) // config.batch_size + 1
    progress_bar = tqdm.tqdm(total=total_steps, desc="step")
    print(f"总训练步数为{total_steps}")

    for step in range(total_steps):
        
        # 1、获取当前批次偏好数据对和拒绝数据对
        preferred_batch_sequences = chosen_input_ids[step*config.batch_size:(step+1)*config.batch_size]
        rejected_batch_sequences = rejected_input_ids[step*config.batch_size:(step+1)*config.batch_size]
        
        # 2、获取当前批次偏好数据对和拒绝数据对的最大长度
        preferred_max_length = max([len(sequence["input_ids"]) for sequence in preferred_batch_sequences])
        rejected_max_length = max([len(sequence["input_ids"]) for sequence in rejected_batch_sequences])

        pad_token_id = tokenizer.pad_token_id

        # 3、分别对偏好数据对和拒绝数据对进行padding
        preferred_padded_sequences = []
        for sequence in preferred_batch_sequences:
            original_sequence_ids = sequence["input_ids"]
            padding_length = preferred_max_length - len(original_sequence_ids)

            padded_seq = torch.nn.functional.pad(
                torch.tensor(original_sequence_ids,dtype=torch.long),
                (0,padding_length),
                mode="constant",
                value=pad_token_id
            ).tolist()

            preferred_padded_sequences.append(padded_seq)
        
        preferred_batch_tensor = torch.tensor(preferred_padded_sequences,dtype=torch.long).to(config.device)
        rejected_padded_sequences = []
        for rejected_sequence in rejected_batch_sequences:
            original_sequence_ids = rejected_sequence["input_ids"]
            padding_length = rejected_max_length - len(original_sequence_ids)

            padded_seq = torch.nn.functional.pad(
                torch.tensor(original_sequence_ids,dtype=torch.long),
                (0,padding_length),
                mode="constant",
                value=pad_token_id
            ).tolist()

            rejected_padded_sequences.append(padded_seq)

        rejected_batch_tensor = torch.tensor(rejected_padded_sequences,dtype=torch.long).to(config.device)
        

        # 4、构建偏好数据对和拒绝数据对的输入和target 张量
        preferred_model_inputs = preferred_batch_tensor[:,:-1]
        preferred_target_labels = preferred_batch_tensor[:,1:]
        rejected_model_inputs = rejected_batch_tensor[:,:-1]
        rejected_target_labels = rejected_batch_tensor[:,1:]

        # 5、计算偏好数据对和拒绝数据对的padding mask和answer mask，并取交集作为最终计算loss的mask
        preferred_padding_mask = torch.where(preferred_target_labels == pad_token_id,0,1)

        rejected_padding_mask = torch.where(rejected_target_labels == pad_token_id,0,1)

        preferred_answer_mask = create_answer_mask(
            preferred_model_inputs,
            tokenizer=tokenizer
        )

        rejected_answer_mask = create_answer_mask(
            rejected_model_inputs,
            tokenizer=tokenizer
        )


        preferred_final_mask = preferred_padding_mask * preferred_answer_mask
        rejected_final_mask = rejected_padding_mask * rejected_answer_mask

        preferred_min_tokens = preferred_final_mask.sum(dim=-1).min().item()
        rejected_min_tokens = rejected_final_mask.sum(dim=-1).min().item()

        if preferred_min_tokens == 0 or rejected_min_tokens == 0:
            print(f"在第{step+1}个step中，偏好数据对或拒绝数据对的answer mask中包含了0个token，跳过该批次")
            progress_bar.update(1)
            skipped_batches_count += 1
            continue

        # 6、偏好数据对前向传播，得到logits
        preferred_logits = model(preferred_model_inputs).logits
        rejected_logits = model(rejected_model_inputs).logits
        
        with torch.no_grad(): # 由于参考模型不参与梯度计算，所以需要在no_grad()上下文管理器中进行前向传播
            reference_preferred_logits = ref_model(preferred_model_inputs).logits.detach() # 参考模型对偏好数据对的logits，需要detach()出来，因为后面需要拿这个结果去计算loss，所以需要从计算图中分离出来
            reference_rejected_logits = ref_model(rejected_model_inputs).logits.detach() # 参考模型对于拒绝数据对的logits计算，类似
        

        # 根据logits来计算log_prob

        step_average_loss = compute_loss(preferred_logits=preferred_logits,
                     rejected_logits=rejected_logits,
                     reference_preferred_logits=reference_preferred_logits,
                     reference_rejected_logits=reference_rejected_logits,
                     preferred_labels=preferred_target_labels,
                     rejected_labels=rejected_target_labels,
                     preferred_answer_mask=preferred_final_mask,
                     rejected_answer_mask=rejected_final_mask,
                     beta=config.beta
                     )
        
        step_average_loss.backward()
        current_learning_rate = cosine_decay(
            step,
            warmup_ratio=config.warmup_ratio,
            total_steps=total_steps,
            max_lr=config.max_lr,
            min_lr=config.min_lr
        )
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate
        
        writer.add_scalar("train_lr",current_learning_rate,step+1)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{step_average_loss.detach().item():.4f}", lr=f"{current_learning_rate:.2e}")
        
        
        training_losses.append(step_average_loss.detach().item())
        
        should_log = (step + 1) % config.logging_iter == 0 or (step + 1) == total_steps
        if should_log:
            recent_losses = training_losses[-config.logging_iter:]
            recent_average_loss = np.nanmean(recent_losses)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(
                f"时间：{current_time} | "
                f"步数：{step+1}/{total_steps} | "
                f"最近{len(recent_losses)}步平均损失：{recent_average_loss:.4f} | "
                f"当前学习率：{current_learning_rate:.2e}"
            )
            writer.add_scalar("train_loss",recent_average_loss,step+1)
    
    print("训练完成")
    print("DPO训练完成")

def save_model(model,tokenizer,output_dir:str):
    """
    保存模型和tokenizer到指定目录
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        output_dir (str): _description_
    """
    import os
    os.makedirs(output_dir,exist_ok=True)   
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"已将模型和tokenizer保存至{output_dir}中")

def main():

    # 配置启动参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_size",type=int,default=10,help="训练数据集的大小")
    parser.add_argument("--batch_size",type=int,default=2,help="批次大小")
    parser.add_argument("--output_dir",type=str,default="./finetuned/Qwen3-0.6B-Instruct-DPO",help="输出目录")
    

    # 解析启动参数
    args = parser.parse_args()
    
    # 1、创建配置实例
    config = DPOTrainConfig(train_data_size=args.train_data_size,batch_size=args.batch_size)

    # 3、获取model和tokenizer
    
    # 4、训练模型
    train(model=model,ref_model=ref_model,tokenizer=tokenizer,config=config)

    save_model(model,tokenizer,output_dir=args.output_dir)

if __name__ == "__main__":
    main()