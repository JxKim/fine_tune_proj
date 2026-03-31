"""
手写SFT训练脚本
"""
from typing import Dict, List
from sympy import Sum
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass

@dataclass
class SFTConfig:

    device:str = "cuda"
    # 学习率
    warmup_ratio:float = 0.1
    min_learing_rate:float = 2e-6
    max_learning_rate:float = 2e-5
    
    batch_size:int = 4

    # 日志相关：
    log_iter:int = 200
    log_dir:str = "logs"

    train_data_size:int = 10000


def test_tokenize_demo():
    """
    使用厂商提供的chat template 将message list转成字符串
    当没有使用chat template时，Base Model推理过程：
        输入：帮我写一个python算法，
        原生LLM输出：，我想要实现冒泡排序，xxxx
    当使用chat template时，经过微调之后的模型推理过程
        输入：帮我写一个python算法，
        Chat Template：<im_start>user帮我写一个python算法<im_end><im_start>assistant
        LLM输出：<im_start>user帮我写一个python算法<im_end><im_start>assistant好的，以下是python实现的xxx

    """
    from transformers import AutoTokenizer
    # 1、获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")
    
    # 2、构造Message List
    message_list = [
        {"role":"user","content":"你好"},
        {"role":"assistant","content":"你好，有什么可以帮你？"}
    ]
    # 3、使用ChatTemplate，将其转换成字符串格式
    res = tokenizer.apply_chat_template(message_list,tokenize=False)

    # 4、在推理阶段，需要使用和训练阶段相同的chat_template，让模型生成
    new_message_list=[
        {"role":"system","content":"你是一个专业的助手"},
        {"role":"user","content":"帮我用python写一个排序算法"}
    ]
    new_res = tokenizer.apply_chat_template(new_message_list,tokenize=False,add_generation_prompt=True)

    print(res)
    print("==============")
    print(new_res)

def create_answer_mask(input_ids,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    """

    
    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵，后面对于assistant回答的部分，将其设置为1
    answer_mask = torch.zeros_like(input_ids)

    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    for idx,ids in enumerate(input_ids):
        # 获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()

        # 排除第一个eos_position: 第一个对应的是system prompt，所以需要将第一个im_end排除
        # <im_start>system
        #你是一个专业的助手<im_end>
        # <im_start>user
        # 什么是习惯？<im_end>
        # <im_start>assistant
        # 习惯是指在一定时间内重复执行的行为。<im_end>
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends

        # <im_start>user
        # 什么是习惯？<im_end>
        # <im_start>assistant
        # 习惯是指在一定时间内重复执行的行为。<im_end>
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

def compute_loss(output_logits,target_labels,answer_mask):
    """
    计算损失函数
    """
    # 1、计算对数概率
    log_probs = torch.nn.functional.log_softmax(output_logits,dim=-1)

    # 2、找到每个token所对应的target label的概率
    target_probs = torch.gather(
        log_probs,
        dim=-1,
        index=target_labels.unsqueeze(-1)
    )

    # 3、获取负对数概率

    negative_target_probs = -target_probs

    negative_target_probs = negative_target_probs.squeeze(-1)

    # 添加掩码

    losses = torch.mul(negative_target_probs,answer_mask)

    # 计算总的token数量
    total_tokens_num = answer_mask.sum()

    # 平均损失：后面会将没有assistant answer的batch过滤掉，所以此处的total_tokens_num一定是大于0的
    avg_loss = losses.sum() / total_tokens_num

    return avg_loss

def test_gather_demo():
    log_probs = torch.tensor(
        [[[0.1,0.2,0.7],[0.3,0.4,0.3]]]
    )
    target_labels = torch.tensor(
        [[0,2]]
    )

    res = torch.gather(
        log_probs,
        dim=-1,
        index=target_labels.unsqueeze(-1)
    )

    print("结果为：",res)

# 3、加载数据集，并对其进行tokenize处理
def get_data_ultrachat_200k(config:SFTConfig,tokenizer)->List[Dict]:
    import datasets
    # 2、下载并处理数据集
    # 2.1 加载UltraChat 200k数据集
    ultrachat_200k_data:datasets.DatasetDict = datasets.load_dataset("./data/ultrachat_200k")
    tokenizer = tokenizer
    train_data = []
    i = 0
    # 2.2 对数据进行tokenize处理
    while True:
        data:List = ultrachat_200k_data["train_sft"][i]["messages"]
        
        data.insert(0,{"role":"system","content":"You are a helpful assistant."})
        # input_ids:{"input_ids":[],"attention_mask":[]}
        input_ids = tokenizer.apply_chat_template(data,tokenize=True,add_generation_prompt=False,truncation=True,max_length=2500)
        
        train_data.append(input_ids)
        i += 1
        if i % 1000 == 0:
            print(f"已经处理了 {i} 条数据")
        # 仅训练train_data_size条数据
        if i == config.train_data_size:
            break
    
    return train_data
def train(model,tokenizer,config:SFTConfig):
    """
    训练脚本
    """
    device = config.device
    model.to(device)
    model.train()
    train_data = get_data_ultrachat_200k(config,tokenizer=tokenizer)

    # 计算有多少个step
    num_steps = len(train_data) // config.batch_size if len(train_data) % config.batch_size == 0 else len(train_data) // config.batch_size + 1

    # 构造optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.max_learning_rate)

    # 构造writer对象
    writer = SummaryWriter(config.log_dir)

    import tqdm
    progress_bar = tqdm.tqdm(total=num_steps)
    pad_token_id = tokenizer.pad_token_id
    for step in range(num_steps):
        # 1、获取当前批次数据，并进行padding处理
        batch_data = train_data[step*config.batch_size:(step+1)*config.batch_size]
        max_len = max( [len(seqs["input_ids"]) for seqs in batch_data])
        padded_data = []
        for ids in batch_data:
            padding_length = max_len - len(ids["input_ids"])

            # padding
            padded_sequence = torch.nn.functional.pad(torch.tensor(ids,dtype=torch.long), (0, padding_length), value=pad_token_id).tolist()
            padded_data.append(padded_sequence)
        
        # 2、将数据转换为tensor
        padded_data = torch.tensor(padded_data,dtype=torch.long).to(device)

        # 3、获取前向传播输入的input_ids和target_labels
        input_ids = padded_data[:,:-1]
        target_labels = padded_data[:,1:]

        # 4、算掩码
        padding_mask = torch.where(input_ids==pad_token_id,0,1)
        answer_mask = create_answer_mask(input_ids,tokenizer)

        final_loss_mask = padding_mask & answer_mask

        # 5、前向传播
        output_logits = model(input_ids).logits

        # 6、计算损失
        loss = compute_loss(output_logits,target_labels,final_loss_mask)

        # 7、反向传播
        loss.backward()

        # 8、学习率更改
        
        optimizer.step()
        optimizer.zero_grad()



if __name__=="__main__":
    test_gather_demo()