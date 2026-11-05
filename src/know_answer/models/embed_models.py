import os

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from typing import Literal, Any


class EmbedLLM:
    def __init__(self, model_provider: Literal["openai", "huggingface"] = "openai",
                 model: str = os.getenv("OPENAI_MODEL_NAME", "huggingface"),
                 **kwargs: Any):
        self.model_provider = model_provider
        self.model = model
        self.embed_agent: Embeddings = init_embeddings(provider=model_provider, model=model, **kwargs)


# --- 客户端使用示例 ---
if __name__ == '__main__':
    llmClient = EmbedLLM("huggingface","BAAI/bge-small-zh-v1.5",cache_folder="D:\\work_store\\huggingface")
    print()
