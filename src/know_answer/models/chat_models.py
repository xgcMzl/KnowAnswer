import os

from langchain.chat_models.base import _ConfigurableModel
from typing import List, Dict, Literal, Any

from langchain.chat_models import init_chat_model



class ChatLLM:
    def __init__(self,
                 model_provider: Literal["deepseek", "openai"] = "deepseek",
                 model: str = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-flash")
                 ):

        self.model_provider = model_provider
        self.model = model
        self.chat_agent: _ConfigurableModel = init_chat_model(model_provider=model_provider, model=model,
                                                              configurable_fields="any")

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.chat_agent.stream(messages)
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     temperature=temperature,
            #     stream=True,
            # )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                # chunk 是一个 AIMessageChunk，其 .content 属性包含当前片段的内容
                content = chunk.content
                print(chunk.content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            print("✅ 大语言模型响应结束:")
            return "".join(collected_content)


        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == '__main__':
    from dotenv import load_dotenv

    # 加载 .env 文件中的环境变量
    load_dotenv()
    try:
        llmClient = ChatLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
