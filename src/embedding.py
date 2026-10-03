from typing import List,Any 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self,model: SentenceTransformer, chunk_size:int=1000,chunk_overlap:int =200):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.model = model
        print(f"[INFO] LOADED EMBEDDING MODEL :{model}")
        
        
    def chunk_documents(self,documents:List[Any]):
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n","\n"," ",""]
            
            )
        chunks=splitter.split_documents(documents)
        print(f"[INFO]Split {len(documents)}documents into {len(chunks)} chunks.")
        return chunks
    
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        embeddings = self.model.encode(
            [chunk.page_content for chunk in chunks],
            show_progress_bar=True
        )

        print(f"[INFO] Created embedding for: {len(embeddings)} chunks")

        return embeddings
    
