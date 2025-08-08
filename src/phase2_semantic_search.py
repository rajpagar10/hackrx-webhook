"""
Phase 2: Semantic Search Module
Handles document embedding generation and semantic search
"""
import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from tqdm import tqdm
import yaml

class SemanticSearchEngine:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize semantic search engine with configuration"""
        self.config = self._load_config(config_path)
        
        # Load embedding model
        model_name = self.config['embeddings']['model_name']
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.vector_db_path = Path(self.config['vector_db']['persist_directory'])
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
        self.collection_name = self.config['vector_db']['collection_name']
        
        # Search settings
        self.top_k = self.config['semantic_search']['top_k']
        self.similarity_threshold = self.config['semantic_search']['similarity_threshold']
        
        # Paths
        self.processed_path = Path(self.config['paths']['processed_data'])

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise

    def generate_embeddings(self, chunked_documents_path: str = None) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks"""
        if chunked_documents_path is None:
            chunked_documents_path = self.processed_path / "chunked_documents.json"
        
        logger.info(f"Loading chunked documents from: {chunked_documents_path}")
        
        # Load chunked documents
        with open(chunked_documents_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text content
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings with progress bar
        batch_size = self.config['embeddings']['batch_size']
        normalize = self.config['embeddings']['normalize']
        
        embeddings = self.model.encode(
            chunk_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Add embeddings to chunk data
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()  # Convert numpy to list for JSON serialization
            embedded_chunks.append(chunk)
        
        # Save embedded chunks
        output_path = self.processed_path / "embedded_chunks.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, indent=2)
        
        logger.success(f"✅ Embeddings generated and saved to: {output_path}")
        return embedded_chunks

    def initialize_vector_database(self, embedded_chunks_path: str = None) -> None:
        """Initialize ChromaDB with embedded chunks"""
        if embedded_chunks_path is None:
            embedded_chunks_path = self.processed_path / "embedded_chunks.json"
        
        logger.info("Initializing vector database...")
        
        # Load embedded chunks
        with open(embedded_chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Prepare data for insertion
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = [chunk['embedding'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                "file_name": chunk['file_name'],
                "chunk_id": chunk['chunk_id'],
                "doc_type": chunk.get('doc_type', 'unknown'),
                "chunk_index": chunk.get('chunk_index', 0),
                "word_count": chunk.get('word_count', 0)
            }
            for chunk in chunks
        ]
        
        # Insert data in batches
        batch_size = 100  # ChromaDB batch size limit
        for i in tqdm(range(0, len(chunks), batch_size), desc="Inserting into ChromaDB"):
            end_idx = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        logger.success(f"✅ Vector database initialized with {len(chunks)} chunks")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into an embedding vector"""
        instruction = self.config['embeddings']['instruction_prefix']
        query_with_instruction = instruction + query
        
        embedding = self.model.encode(
            query_with_instruction,
            normalize_embeddings=self.config['embeddings']['normalize']
        )
        
        return embedding

    def semantic_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform semantic search for a query"""
        if top_k is None:
            top_k = self.top_k
        
        # Ensure collection is loaded
        if not hasattr(self, 'collection'):
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except:
                logger.error(f"Collection '{self.collection_name}' not found. Please initialize the database first.")
                return []
        
        logger.info(f"Searching for: '{query}'")
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            result = {
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "rank": i + 1
            }
            
            # Filter by similarity threshold
            if result['similarity_score'] >= self.similarity_threshold:
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} relevant results")
        return formatted_results

    def search_interactive(self):
        """Interactive search interface"""
        logger.info("🔍 Starting interactive search mode...")
        logger.info("Type 'quit' to exit")
        
        while True:
            try:
                query = input("\\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("Exiting interactive search...")
                    break
                
                if not query:
                    continue
                
                # Perform search
                results = self.semantic_search(query)
                
                if not results:
                    print("No relevant results found.")
                    continue
                
                # Display results
                print(f"\\n🔍 Found {len(results)} relevant results for: '{query}'\\n")
                
                for result in results:
                    metadata = result['metadata']
                    score = result['similarity_score']
                    
                    print(f"📄 Rank {result['rank']} (Score: {score:.4f})")
                    print(f"   Source: {metadata['file_name']} | Chunk: {metadata['chunk_id']}")
                    print(f"   Text: {result['text'][:300]}...")
                    print()
                    
            except KeyboardInterrupt:
                logger.info("\\nExiting interactive search...")
                break
            except Exception as e:
                logger.error(f"Search error: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection"""
        if not hasattr(self, 'collection'):
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except:
                return {"error": "Collection not found"}
        
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.config['embeddings']['model_name'],
            "similarity_metric": self.config['vector_db']['similarity_metric']
        }

    def build_complete_search_index(self) -> None:
        """Complete pipeline: generate embeddings and initialize database"""
        logger.info("🚀 Building complete search index...")
        
        # Step 1: Generate embeddings
        embedded_chunks = self.generate_embeddings()
        
        # Step 2: Initialize vector database
        self.initialize_vector_database()
        
        # Step 3: Display statistics
        stats = self.get_collection_stats()
        logger.success(f"✅ Search index built successfully!")
        logger.info(f"   Collection: {stats['collection_name']}")
        logger.info(f"   Total chunks: {stats['total_chunks']}")
        logger.info(f"   Model: {stats['embedding_model']}")


def main():
    """Main function to run semantic search setup"""
    search_engine = SemanticSearchEngine()
    
    try:
        # Build complete search index
        search_engine.build_complete_search_index()
        
        # Test with sample queries
        sample_queries = [
            "What is the grace period for premium payment?",
            "Is cataract surgery covered?",
            "What are the waiting periods for pre-existing diseases?"
        ]
        
        print("\\n🧪 Testing with sample queries...")
        for query in sample_queries:
            print(f"\\nQuery: {query}")
            results = search_engine.semantic_search(query, top_k=3)
            
            for result in results[:2]:  # Show top 2 results
                metadata = result['metadata']
                print(f"  📄 {metadata['file_name']} (Score: {result['similarity_score']:.3f})")
                print(f"     {result['text'][:150]}...")
        
        # Optional: Start interactive mode
        print("\\n" + "="*50)
        response = input("Start interactive search mode? (y/n): ")
        if response.lower().startswith('y'):
            search_engine.search_interactive()
        
    except Exception as e:
        logger.error(f"Error in semantic search setup: {e}")
        raise


if __name__ == "__main__":
    main()