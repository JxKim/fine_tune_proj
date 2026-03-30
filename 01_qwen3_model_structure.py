import torch
import torch.nn as nn
from pathlib import Path
import re
# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # 词表大小
    "context_length": 40_960,  # 训练时使用的上下文长度
    "emb_dim": 1024,           # 嵌入维度
    "n_heads": 16,             # 注意力头数
    "n_layers": 28,            # 层数
    "hidden_dim": 3072,        # 中间层维度
    "head_dim": 128,           # GQA头维度
    "qk_norm": True,           # 是否需要对Query和Key进行归一化
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 输入层：token_embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # transformer block
        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 没有传入head_dim时，默认使用emb_dim // n_heads作为head_dim
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        # 获取到每个位置的每组向量的旋转的余弦值和正弦值
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # 追踪KV Cache 当前位置的索引

    def forward(self, in_idx, cache=None):
        # 前向传播

        # 输入层：获取到token_embedding
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        # 输出前先进行层归一化
        x = self.final_norm(x)
        # 输出层
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # GQA注意力层
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # 保存输入，用于后面进行残差连接（GQA的残差连接）
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # 进行残差连接

        # 保存输入，用于后面进行残差连接（FFN的残差连接）
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # 进行残差连接

        return x, next_cache

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Query层参数为全量的：d_in输入，d_out输出
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        
        # Key和Value层参数不是全量的，d_in输入，num_kv_groups * head_dim输出
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # 获取Q, K, V
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Queries分成num_heads个头，每个头的维度为head_dim
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Keys和Values分成num_kv_groups个组，每个组的维度为head_dim
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 额外的归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # 重点：对queries和keys应用旋转位置编码
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # reset RoPE
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # 复制Keys和Values，使得在group_size中，所有的Query都共用相同的Keys和Values
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 注意力得分计算
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    计算每个位置的余弦值和正弦值，用于旋转位置编码
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 计算位置索引：[0,1,2,3,...,context_length-1] shape: (context_length,)
    positions = torch.arange(context_length, dtype=dtype)

    # 计算每个位置的每组旋转的角度
    # positions.unsqueeze(1): (context_length,1)
    # inv_freq.unsqueeze(0): (1,head_dim // 2)
    # angles: (context_length, head_dim // 2)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # 将angles扩展到head_dim维度，shape: (context_length, head_dim)
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # 预计算每个角度的余弦值和正弦值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    """
    使用计算好的cos和sin来计算旋转位置编码之后的x，注意，此处获取分组向量不是使用紧挨着的两个值构造而成的，
    如果 head_dim = 8，此处的配对关系就是：
                    (0, 4)
                    (1, 5)
                    (2, 6)
                    (3, 7)
    而不是常见的：
                    (0, 1)
                    (2, 3)
                    (4, 5)
                    (6, 7)
    Args:
        x (_type_): 需要应用旋转位置编码的张量
        cos (_type_): 预计算好的余弦值张量
        sin (_type_): 预计算好的正弦值张量
    Returns:
        旋转后的张量
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 将x切分为前一半和后面一半
    x1 = x[..., : head_dim // 2]  # 前面一半：Shape: (batch_size, num_heads, seq_len, head_dim // 2)
    x2 = x[..., head_dim // 2:]  # 后面一半：Shape: (batch_size, num_heads, seq_len, head_dim // 2)

    # 调整sin和cos的形状：当前这段 token 取出对应位置的 cos 和 sin，并调整 shape 以便广播
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1) # Shape: (batch_size, num_heads,seq_len, head_dim)
    # x*cos = [x1*cos ,x2*cos]
    # rotated*sin = [-x2*sin ,x1*sin]
    # x_rotated = [x1*cos-x2*sin, x1*sin+x2*cos]
    # 旋转矩阵 [cos, -sin; sin, cos], x_rotated = [x1,x2]*旋转矩阵
    x_rotated = (x * cos) + (rotated * sin)

    # 返回旋转后的x
    return x_rotated.to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # insert no <think> tag, just a new line
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s

def generate_text(input_ids,model:Qwen3Model,tokenizer:Qwen3Tokenizer):

    import torch

    with torch.no_grad():
        output_logits = model(input_ids)
        # 使用贪心策略，
        res = output_logits.argmax(dim=-1) # shape:batch_size,seq_len
    
    res_list = res.tolist()
    print(res_list)
    res = tokenizer.decode(res_list[0])
    print(res)
    return res

def main():
    """
    测试Qwen3模型的生成
    Returns:
        _type_: _description_
    """
    import torch
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=r"Qwen3-0.6B\tokenizer.json")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.eval()
    model.to("cuda")
    input_ids = torch.tensor(tokenizer.encode("你好，今天天气真好啊")).unsqueeze(0).to("cuda")
    output = generate_text(input_ids,model,tokenizer)
    
    print(output)

if __name__ == "__main__":
    main()
