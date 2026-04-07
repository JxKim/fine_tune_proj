

def openai_demo():
    from openai import OpenAI

    # 连接本地
    client = OpenAI(
        base_url="http://localhost:8000/v1/",
        api_key="none"  # 占位符，可忽略
    )

    # 多轮对话
    response = client.chat.completions.create(
        model="Qwen3-0.6B",  # 指定模型,必须与启动vllm时指定的名字一致
        messages=[
            # {"role": "user", "content": "我的猫死了，我很难过"}
            {"role": "user", "content": "什么是langchain"}
        ],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}  # 关键参数
        }
    )

    # print(response)
    print(response.choices[0].message.content)

def langchain_demo():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(
        model="Qwen3-0.6B",
        base_url="http://localhost:8000/v1/",
        api_key="none",
        temperature=0)

    # 构建消息（支持多角色）
    messages = [
        SystemMessage(content="你是一个专业的技术助手，回答需简洁准确"),
        HumanMessage(content="LangChain如何调用OpenAI风格的大模型？")
    ]

    # 调用模型获取响应
    response = llm.invoke(messages)
    print(response.content)

langchain_demo()
