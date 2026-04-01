import time
import tiktoken
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 获取到每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 用来合并多头信息的线性层
        self.dropout = nn.Dropout(dropout)
        # 将下三角掩码方阵保存在显存中冻结，形状：(SeqLen, SeqLen)
        # 得到的结果为：
        # tensor([[0., 1., 1.,  ..., 1., 1., 1.],
        #         [0., 0., 1.,  ..., 1., 1., 1.],
        #         [0., 0., 0.,  ..., 1., 1., 1.],
        #         ...,
        #         [0., 0., 0.,  ..., 0., 1., 1.],
        #         [0., 0., 0.,  ..., 0., 0., 1.],
        #         [0., 0., 0.,  ..., 0., 0., 0.]])
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

        # 将kv cache保存在显存中冻结，后面可以通过self.cache_k / self.cache_v访问，初始值为None，persistent=False表示不保存到模型的state_dict中
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        

    def forward(self, x, use_cache=False):
        
        batch_size, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (batch_size, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # 通过添加一个num_heads维度来隐式地分割矩阵   
        # keys_new[0,0,0,:]   表示的是：第一个序列的第一个token的第一个头的key向量  
        keys_new = keys_new.view(batch_size, num_tokens, self.num_heads, self.head_dim) # Shape: (batch_size,num_tokens,num_heads,head_dim)
        values_new = values_new.view(batch_size, num_tokens, self.num_heads, self.head_dim) # Shape: (batch_size,num_tokens,num_heads,head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim) # Shape: (batch_size,num_tokens,num_heads,head_dim)

        if use_cache:
            # 使用kv cache时
            if self.cache_k is None:
                # 第一次输入整个prompt时，
        # 会计算整个prompt序列里面所有token的，K，V，Q
                # 如果kv cache为空,说明是prefill阶段，
                # 将计算得到的kv cache存储下来
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                # 后面的每个token前向传播
                # 如果kv cache不为空，说明是decode阶段
                # 说明来到的是预测出的下一个token，那么将这个token的kv拼接到kv cache中
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1) # Shape: (batch_size,num_tokens,num_heads,head_dim)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1) # Shape: (batch_size,num_tokens,num_heads,head_dim)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        

        # Transpose: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数：
        # keys.transpose(2, 3) -> (batch_size,num_heads,head_dim,num_tokens)
        # 得到的attn_scores的shape为(batch_size,num_heads,num_tokens,num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  


        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            # 用 cache 时，K 包含历史，Q 是当前新来的 token，所以行不能再从 0 开始切，要从 ptr_current_pos 开始切
            # 第一次输入整个prompt，假设seq_len = 20：预填充prefill，self.ptr_current_pos = 0, num_tokens_Q = 20
            # num_tokens_k = 20
            # self.mask.bool():
        # tensor([[False., True., True.,  ..., True., True., True.],
        #         [False., False., True.,  ..., True., True., True.],
        #         [False., False., False.,  ..., True., True., True.],
        #         ...,
        #         [0., 0., 0.,  ..., 0., 1., 1.],
        #         [0., 0., 0.,  ..., 0., 0., 1.],
        #         [0., 0., 0.,  ..., 0., 0., 0.]])
        # 通过切片，prefill阶段，取前20行，前20列的mask_bool

            # 后面的单个token前向传播阶段：Decode(这个概念是和prefill对应的，这是KV Cache里面的两个概念)
            # self.ptr_current_pos = 20, num_tokens_Q = 1,num_tokens_K=21
            # mask_bool : 从self.mask里面取第20行，前21列的mask_bool
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            # ptr_current_pos更新，这个表示当前处理到的位置
            self.ptr_current_pos += num_tokens_Q
        # 不用cache时，直接获取num_tokens_Q个token的mask_bool
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # 把未来位置置成 -inf
        # w_11,负无穷。。。
        # w_21,w_22,负无穷，负无穷。。。 
        # 第一行 [False, True, True, ..., True, True, True]
        # 填充完：[0.12,-inf,...,-inf,-inf,-inf]

        # 第二行：[False,False,True,...,True,True,True]
        # 填充完：[0.12,0.34,...,-inf,-inf,-inf,-inf]
        # ... 以此实现了带掩码的自注意力机制
        # decode阶段：取第21行，前21列:[False , False, False,....  False]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 用注意力权重对V加权求和，再合并多头
       
        # attn_weights.shape: (batch_size, num_heads, num_tokens, num_tokens)
        # values.shape: (batch_size, num_heads, num_tokens, head_dim)
        # 得到的结果为：(batch_size, num_heads, num_tokens, head_dim)
         # transpose之后的结果为： (batch_size,num_tokens,num_heads,head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        
        # 此处调用contiguous()是为了确保context_vec的内存是连续的，否则在view操作中会报错
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        # 合并多头信息，并经过out_proj进行线性融合，得到最终输出, self.d_out = self.num_heads * self.head_dim
        context_vec = self.out_proj(context_vec)  

        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# Transformer Block层：
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 多头注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        
        # 先保存原始输入，后面用于残差连接（多头注意力层后的残差连接）
        shortcut = x
        # 层归一化（Pre LayerNorm）
        x = self.norm1(x)

        # 多头注意力前向传播
        x = self.att(x, use_cache=use_cache)
        

        # DropOut层
        x = self.drop_shortcut(x)
        # 残差连接
        x = x + shortcut  

        # 先保存原始输入，后面用于残差连接（FFN后的残差连接）
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # 残差连接
        x = x + shortcut  

        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 输入层：token embedding + position embedding
        # self.weight = Parameter(
            #     torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
            #     requires_grad=not _freeze,
            # )
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer Block层: 由多层transformer block堆叠而成
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 输出层：将向量映射成词表大小
        # self.weight = Parameter(
        #     torch.empty((cfg["vocab_size"], cfg["emb_dim"]), **factory_kwargs)
        # )
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 输出层的权重与token embedding的权重共享
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx, use_cache=False):
        # seq_len:20
        batch_size, seq_len = in_idx.shape
        # batch_size,seq_len,emb_dim
        tok_embeds = self.tok_emb(in_idx)

        if use_cache:
            # 第一次将整个prompt输入：此时self.current_pos=0
            # pos_ids: [0,1,2,...,18,19]
            # 后面进行逐个token前向传播时，
            # 第二次pos_ids:[20]，第三次pos_ids:[21] ...
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            # 接下来，我们对current_pos更新，self.current_pos = 0+20 =20
            self.current_pos += seq_len
        else:
            # [0,1,2,...,18,19]
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        # [0_position_embedding,1_position_embedding,...,19_position_embedding,] 
        # [[0_position_embedding,1_position_embedding,...,19_position_embedding,]]
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # token_embeedding:shape:[batch_size, seq_len]+[1,seq_len]
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # transformer block层前向传播
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        
        # 最后的层归一化
        x = self.final_norm(x)

        # 输出层
        # shape: batch_size, seq_len, vocab_size
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0

def generate_text_simple_cached(model:GPTModel, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings # 上下文长度，如果没有指定，使用模型的位置编码的长度

    with torch.no_grad():
        if use_cache:
            print("使用KV Cache 进行生成")
            # 使用完整的prompt初始化KV Cache
            model.reset_kv_cache() # 清空kv cache，因为新的提示词到来了
            # 预填充（prefill），计算prompt的kv，然后缓存下来，注意！输入的是整个提示词
            # 假设序列里面有10个token：这个时候，10个token的K,V向量，都已经算完了，这个时候可以在模型内存把这些向量缓存起来
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # 贪婪策略：直接取出概率最大的下一个token
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 将其拼接到历史token列表后面
                # idx:表示的是历史整个序列
                idx = torch.cat([idx, next_idx], dim=1)
                # 给模型输入最新的下一个token，注意！只输入了一个token，因为模型已经缓存了之前的KV
                logits = model(next_idx, use_cache=True)
        else:
            print("不使用KV Cache 进行生成")
            for _ in range(max_new_tokens):
                # 输入prompt: idx 156k tokens, ctx_len:128k ，通过-ctx_len: 就能够取到最后的128k个token
                logits = model(idx[:, -ctx_len:], use_cache=False)
                # 去输出的最后一个作为下一个token
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 将token拼接到历史token列表后面，注意！整个token列表都被输入给了模型
                idx = torch.cat([idx, next_idx], dim=1)

    return idx

def main():
    # GPT2 模型配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # 层数量
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # qkv三个线性层是否使用偏置项
    }

    torch.manual_seed(123) # 固定随机种子
    model = GPTModel(GPT_CONFIG_124M) # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 切换到评估模式，关闭dropout

    start_context = "Hello, I am a student. I am working on a project.Do you think I should continue?"

    # 此处通过tiktoken获取到gpt2对应的tokenizer，用于将文本编码为token id
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    # 进行文本生成，获取到生成的token ids
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=500,
        use_cache=False
    )
    
    total_time = time.time() - start

    # 将生成的token ids解码为文本
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if __name__ == "__main__":
    main()