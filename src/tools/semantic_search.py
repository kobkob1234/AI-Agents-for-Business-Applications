import os
import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.tools.vector_store_manager import VectorStoreManager

class SemanticSearch:
    def __init__(self, db_dir: str = "chroma_db", collection_name: str = "asrs_reports"):
        print("Initializing Semantic Search Tool...")
        # db_dir is legacy if we use manager defaults
        self.manager = VectorStoreManager(index_name=collection_name)
        self.vectorstore = self.manager.get_vector_store()
        self.embeddings = self.manager.embeddings
        
    def ingest_data(self, df: pd.DataFrame, batch_size: int = 100):
        """
        Ingests data from the DataFrame into ChromaDB.
        Assumes df has 'ACN' and 'Full_Narrative' columns.
        """
        print(f"Ingesting {len(df)} documents into {self.manager.index_name}...")
        
        documents = []
        for _, row in df.iterrows():
            if 'Full_Narrative' not in row or not isinstance(row['Full_Narrative'], str) or not row['Full_Narrative'].strip():
                continue
                
            # Metadata: Store relevant fields for filtering later if needed
            metadata = {
                "ACN": str(row.get('ACN', '')),
                "Date": str(row.get('Date', '')),
                "Make_Model": str(row.get('Make Model Name', ''))
            }
            
            doc = Document(
                page_content=row['Full_Narrative'],
                metadata=metadata
            )
            documents.append(doc)
            
        # Ingest in batches
        total_docs = len(documents)
        print(f"Prepared {total_docs} valid documents.")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            self.vectorstore.add_documents(batch)
            print(f"Index: Sampled {min(i+batch_size, total_docs)}/{total_docs}", end='\r')
            
        print("\nIngestion complete.")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search.
        """
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "ACN": doc.metadata.get("ACN"),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        return results

if __name__ == "__main__":
    # Test
    try:
        tool = SemanticSearch()
        print("Initialized Semantic Search.")
        # Minimal test ingest
        if os.environ.get("OPENAI_API_KEY") is None:
             # Ingest a fake doc to search
             tool.vectorstore.add_documents([Document(page_content="Test fatigue report", metadata={"ACN":"TEST001"})])
             
        results = tool.search("pilot fatigue related to scheduling")
        print(f"Found {len(results)} results.")
        for r in results:
            print(f" - {r['ACN']}: {r['content'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")
