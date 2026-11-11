from llama_index.core import SimpleDirectoryReader

from know_answer.models.embed_models import EmbedLLM

documents = SimpleDirectoryReader("D:\\ai_project\\KnowAnswer\\data").load_data()
print(f"loaded {len(documents)} document(s)")
print("first document preview:", documents[0].text[:200])

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=50,
    chunk_overlap=0,
)
nodes = splitter.get_nodes_from_documents(documents)
print(f"split into {len(nodes)} node(s)")
for i, node in enumerate(nodes[:3]):
    print(f"--- node {i} (len={len(node.text)}) ---")
    print(node.text[:200])


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model

# 验证维度
test_vec = embed_model.get_text_embedding("测试文本")
print("embedding dimension =", len(test_vec))


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    uri="http://106.55.165.169:19530",
    collection_name="demo_semantic_search",
    dim=512,                 # 必须与第 4 步确认的维度一致
    overwrite=True,          # demo 阶段方便重复运行；生产环境谨慎使用
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)
print("index built and persisted to Milvus")