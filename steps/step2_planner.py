import os


from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

topic = input("请输入调研主题：")

prompt = f"""你是一名资深研究规划专家。
用户的调研主题是：{topic}

请把这个主题拆解成 3 个具体、可独立调研的子问题。

要求：
- 只输出 3 行，每行一个子问题
- 不要编号，不要解释，不要任何其他内容
"""

print("\n正在调用大模型...\n")
resp = llm.invoke(prompt)

print("=" * 55)
print("模型原始输出：")
print("=" * 55)
print(resp.content)

questions = []
for line in resp.content.strip().split("\n"):
    q = line.strip().lstrip("0123456789.、）- ").strip()
    if q:
        questions.append(q)
print("\n" + "=" * 55)
print("解析后的子问题：")
print("=" * 55)
for i, q in enumerate(questions[:3], 1):
    print(f"{i}. {q}")

