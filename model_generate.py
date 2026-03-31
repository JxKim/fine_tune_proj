from transformers import AutoModelForCausalLM,AutoTokenizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--prompt",type=str)
parser.add_argument("--model_path",type=str)
args = parser.parse_args()



# 1、加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model.eval()
model.to("cuda")

# 2、对prompt进行tokenize
prompt = args.prompt
# 为什么要传入add_generation_prompt = True，是因为我们的掩码当中，对于生成<im_start>assistant\n，这三个token我们没有计算损失
# 没有计算损失，就意味着模型没有学习到这三个分布，它学习的是如何沿着这三个token回答
message_list = [ {"role":"system","content":"你是一个专业的助手"},{"role":"user","content":prompt}]
inputs = tokenizer.apply_chat_template(message_list,tokenize = True,add_generation_prompt = True,return_tensors = "pt")
# 调用generate方法，进行自回归生成
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
output_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask,eos_token_id=[151643,151645],max_new_tokens=128)
res_token_ids = output_ids[0][len(input_ids[0]):]
res = tokenizer.decode(res_token_ids)
output_text = res
print(output_text)

