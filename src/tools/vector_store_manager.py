import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings

# Try importing Pinecone, handle if missing
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

class VectorStoreManager:
    def __init__(self, index_name="asrs-reports"):
        self.index_name = index_name
        app_env = os.environ.get("APP_ENV", "").strip().lower()
        strict_flag = os.environ.get("REQUIRE_STRICT_STACK", "").strip().lower()
        self.strict_mode = app_env in {"prod", "production"} or strict_flag in {"1", "true", "yes"}
        self.embeddings = self._get_embeddings()

    def _get_embeddings(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
        model_name = "RPRTHPB-text-embedding-3-small"

        if self.strict_mode and "llmod.ai" not in base_url:
            raise RuntimeError("Strict mode: OPENAI_BASE_URL must point to LLMod.ai")

        if api_key:
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            if self.strict_mode:
                raise RuntimeError("Strict mode: OPENAI_API_KEY is required for embeddings")
            print("Using FakeEmbeddings")
            return FakeEmbeddings(size=1536)

    def get_vector_store(self):
        pinecone_key = os.environ.get("PINECONE_API_KEY")

        if pinecone_key and HAS_PINECONE:
            print(f"Connecting to Pinecone Index: {self.index_name}")
            try:
                self.create_pinecone_index_if_needed()
            except Exception as e:
                if self.strict_mode:
                    raise RuntimeError(f"Strict mode: Pinecone index setup failed: {e}") from e
                print(f"Warning: Failed to ensure Pinecone index exists: {e}")

            return PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )

        if self.strict_mode:
            reasons = []
            if not pinecone_key:
                reasons.append("missing PINECONE_API_KEY")
            if not HAS_PINECONE:
                reasons.append("pinecone SDK not installed")
            raise RuntimeError("Strict mode: Pinecone is required; " + ", ".join(reasons))

        print("Using Local ChromaDB")
        return Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings,
            collection_name="asrs-reports"
        )

    def create_pinecone_index_if_needed(self):
        if not (os.environ.get("PINECONE_API_KEY") and HAS_PINECONE):
            if self.strict_mode:
                raise RuntimeError("Strict mode: cannot create Pinecone index without key and SDK")
            print("Skipping Pinecone creation (No Key or Lib)")
            return

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        import time

        if self.index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index {self.index_name}...")
            pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Pinecone Index Ready.")
        else:
            print("Pinecone Index already exists.")
