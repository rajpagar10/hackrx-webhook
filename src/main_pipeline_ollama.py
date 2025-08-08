"""
Main Pipeline - Updated for Open Source LLMs (Ollama)
Orchestrates all phases using local LLM instead of OpenAI
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import yaml
from datetime import datetime

# Import all phase modules (updated imports)
from phase1_document_processing import DocumentProcessor
from phase2_semantic_search import SemanticSearchEngine
from phase3_query_processing import QueryProcessor
from phase4_ollama_llm import OllamaLLMAnalyzer, PolicyAnalysis


@dataclass
class QueryResponse:
    """Final structured response for user queries"""
    query: str
    decision: str  # "approved", "rejected", "requires_review"
    amount: Optional[float]
    justification: str
    relevant_clauses: List[Dict[str, Any]]
    query_analysis: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: str
    llm_provider: str = "ollama"  # Track which LLM was used

class LLMDocumentProcessor:
    """Main pipeline orchestrating all processing phases with Ollama LLM"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the complete processing pipeline with open source LLM"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize all phase processors
        logger.info("🚀 Initializing LLM Document Processing System (Open Source)...")
        
        try:
            self.document_processor = DocumentProcessor(config_path)
            self.search_engine = SemanticSearchEngine(config_path)
            self.query_processor = QueryProcessor(config_path) 
            self.llm_analyzer = OllamaLLMAnalyzer(config_path)  # Using Ollama now
            
            # Verify LLM connection
            if self.llm_analyzer.test_connection():
                logger.success("✅ All processors initialized successfully (Ollama connected)")
            else:
                logger.warning("⚠️ Processors initialized but Ollama connection failed")
                logger.info("💡 Make sure Ollama is running: ollama serve")
            
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def _load_config(self) -> Dict:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise

    def setup_system(self, rebuild_index: bool = False) -> bool:
        """
        Set up the complete document processing system
        
        Args:
            rebuild_index: Whether to rebuild the search index from scratch
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("⚙️ Setting up LLM Document Processing System (Open Source)...")
            
            # Phase 1: Process documents
            if rebuild_index or not self._check_processed_documents():
                logger.info("📄 Phase 1: Processing documents...")
                self.document_processor.process_all_documents()
                logger.success("✅ Phase 1 complete")
            else:
                logger.info("📄 Phase 1: Using existing processed documents")
            
            # Phase 2: Build search index
            if rebuild_index or not self._check_search_index():
                logger.info("🔍 Phase 2: Building search index...")
                self.search_engine.build_complete_search_index()
                logger.success("✅ Phase 2 complete")
            else:
                logger.info("🔍 Phase 2: Using existing search index")
            
            # Phase 3 & 4: Verify LLM connection
            if self.llm_analyzer.test_connection():
                logger.success("🤖 Phase 4: Ollama LLM ready")
            else:
                logger.warning("⚠️ Phase 4: Ollama LLM not connected")
                logger.info("💡 Start Ollama: ollama serve")
                logger.info("💡 Download model: ollama pull llama3")
            
            logger.success("🎉 System setup complete!")
            return True
            
        except Exception as e:
            logger.error(f"System setup failed: {e}")
            return False

    def _check_processed_documents(self) -> bool:
        """Check if documents have been processed"""
        processed_path = Path(self.config['paths']['processed_data'])
        required_files = ['chunked_documents.json', 'embedded_chunks.json']
        
        return all((processed_path / file).exists() for file in required_files)

    def _check_search_index(self) -> bool:
        """Check if search index exists"""
        vector_db_path = Path(self.config['vector_db']['persist_directory'])
        return vector_db_path.exists() and any(vector_db_path.iterdir())

    def process_query(self, raw_query: str) -> QueryResponse:
        """
        Process a single query through the complete pipeline using Ollama
        
        Args:
            raw_query: User's natural language query
            
        Returns:
            QueryResponse: Structured response with decision and justification
        """
        start_time = datetime.now()
        logger.info(f"🔍 Processing query with Ollama: '{raw_query[:100]}...'")
        
        try:
            # Phase 3: Query Processing
            logger.debug("Phase 3: Processing and analyzing query...")
            query_result = self.query_processor.process_query(raw_query)
            
            # Phase 2: Semantic Search  
            logger.debug("Phase 2: Performing semantic search...")
            search_results = self.search_engine.semantic_search(
                query_result['canonical_query'],
                top_k=self.config['semantic_search']['top_k']
            )
            
            # Phase 4: Ollama LLM Analysis
            logger.debug("Phase 4: Analyzing with Ollama LLM...")
            analysis = self.llm_analyzer.analyze_query(
                query=raw_query,
                entities=query_result['entities'],
                retrieved_chunks=search_results
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create final response
            response = QueryResponse(
                query=raw_query,
                decision=analysis.decision.lower(),
                amount=analysis.amount,
                justification=analysis.reasoning,
                relevant_clauses=analysis.relevant_clauses,
                query_analysis={
                    "category": query_result['category'],
                    "entities": query_result['entities'],
                    "matched_keywords": query_result['matched_keywords']
                },
                confidence=analysis.confidence,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                llm_provider=f"ollama-{self.llm_analyzer.model}"
            )
            
            logger.success(f"✅ Query processed - Decision: {response.decision}, Time: {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Return error response
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                query=raw_query,
                decision="error",
                amount=None,
                justification=f"Processing failed due to system error: {str(e)}. Please check Ollama connection and try again.",
                relevant_clauses=[],
                query_analysis={},
                confidence=0.0,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                llm_provider="ollama-error"
            )

    def process_batch_queries(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries in batch"""
        logger.info(f"📋 Processing {len(queries)} queries in batch with Ollama...")
        
        responses = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            response = self.process_query(query)
            responses.append(response)
        
        logger.success(f"✅ Batch processing complete - {len(responses)} queries processed")
        return responses

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics including Ollama status"""
        try:
            stats = {
                "system_status": "operational",
                "timestamp": datetime.now().isoformat(),
                "llm_provider": "ollama",
                "components": {}
            }
            
            # Document processor stats
            try:
                processed_path = Path(self.config['paths']['processed_data'])
                if (processed_path / "chunked_documents.json").exists():
                    with open(processed_path / "chunked_documents.json", 'r') as f:
                        chunks = json.load(f)
                    
                    stats["components"]["documents"] = {
                        "total_chunks": len(chunks),
                        "unique_files": len(set(chunk['file_name'] for chunk in chunks)),
                        "status": "ready"
                    }
                else:
                    stats["components"]["documents"] = {"status": "not_processed"}
            except Exception as e:
                stats["components"]["documents"] = {"status": "error", "error": str(e)}
            
            # Search engine stats
            try:
                search_stats = self.search_engine.get_collection_stats()
                stats["components"]["search_index"] = {
                    "total_vectors": search_stats.get("total_chunks", 0),
                    "embedding_model": search_stats.get("embedding_model", "unknown"),
                    "status": "ready" if search_stats.get("total_chunks", 0) > 0 else "empty"
                }
            except Exception as e:
                stats["components"]["search_index"] = {"status": "error", "error": str(e)}
            
            # Ollama LLM status
            try:
                model_info = self.llm_analyzer.get_model_info()
                stats["components"]["llm_analyzer"] = {
                    "provider": "ollama",
                    "model": model_info.get("current_model", "unknown"),
                    "base_url": model_info.get("base_url", "unknown"),
                    "status": model_info.get("status", "unknown"),
                    "available_models": model_info.get("available_models", [])
                }
                
                if model_info.get("status") == "disconnected":
                    stats["components"]["llm_analyzer"]["error"] = model_info.get("error", "Connection failed")
                    
            except Exception as e:
                stats["components"]["llm_analyzer"] = {
                    "provider": "ollama",
                    "status": "error", 
                    "error": str(e)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "llm_provider": "ollama"
            }

    def test_system(self) -> Dict[str, Any]:
        """Run comprehensive system tests with Ollama"""
        logger.info("🧪 Running system tests with Ollama LLM...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "llm_provider": "ollama",
            "tests": {}
        }
        
        # Test queries
        test_queries = self.config['development']['sample_queries']
        
        try:
            # Test each component
            logger.info("Testing document processing...")
            test_results["tests"]["document_processing"] = self._test_document_processing()
            
            logger.info("Testing semantic search...")
            test_results["tests"]["semantic_search"] = self._test_semantic_search()
            
            logger.info("Testing query processing...")
            test_results["tests"]["query_processing"] = self._test_query_processing(test_queries[0])
            
            logger.info("Testing Ollama LLM connection...")
            test_results["tests"]["ollama_connection"] = self._test_ollama_connection()
            
            logger.info("Testing LLM analysis...")
            test_results["tests"]["llm_analysis"] = self._test_llm_analysis()
            
            # Test end-to-end pipeline
            logger.info("Testing end-to-end pipeline...")
            test_results["tests"]["end_to_end"] = self._test_end_to_end(test_queries[0])
            
            # Determine overall status
            all_passed = all(
                test.get("status") == "passed" 
                for test in test_results["tests"].values()
            )
            
            test_results["overall_status"] = "passed" if all_passed else "failed"
            
            logger.success(f"✅ System tests complete - Status: {test_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            test_results["overall_status"] = "error"
            test_results["error"] = str(e)
        
        return test_results

    def _test_ollama_connection(self) -> Dict[str, Any]:
        """Test Ollama connection specifically"""
        try:
            if self.llm_analyzer.test_connection():
                model_info = self.llm_analyzer.get_model_info()
                return {
                    "status": "passed", 
                    "message": f"Connected to Ollama with model: {model_info.get('current_model', 'unknown')}",
                    "model_info": model_info
                }
            else:
                return {
                    "status": "failed", 
                    "error": "Cannot connect to Ollama",
                    "suggestions": [
                        "Start Ollama: ollama serve",
                        "Check if port 11434 is available",
                        "Download model: ollama pull llama3"
                    ]
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _test_document_processing(self) -> Dict[str, Any]:
        """Test document processing component"""
        try:
            processed_path = Path(self.config['paths']['processed_data'])
            required_files = ['chunked_documents.json', 'embedded_chunks.json']
            
            missing_files = [f for f in required_files if not (processed_path / f).exists()]
            
            if missing_files:
                return {
                    "status": "failed", 
                    "error": f"Missing files: {missing_files}"
                }
            
            return {"status": "passed", "message": "All processed files exist"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _test_semantic_search(self) -> Dict[str, Any]:
        """Test semantic search component"""
        try:
            results = self.search_engine.semantic_search("test query", top_k=1)
            
            if not results:
                return {"status": "failed", "error": "No search results returned"}
            
            return {
                "status": "passed", 
                "message": f"Search returned {len(results)} results"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _test_query_processing(self, test_query: str) -> Dict[str, Any]:
        """Test query processing component"""
        try:
            result = self.query_processor.process_query(test_query)
            
            required_fields = ['normalized_query', 'canonical_query', 'category', 'entities']
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                return {"status": "failed", "error": f"Missing fields: {missing_fields}"}
            
            return {"status": "passed", "message": f"Query processed - Category: {result['category']}"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _test_llm_analysis(self) -> Dict[str, Any]:
        """Test Ollama LLM analysis component"""
        try:
            # Simple test with minimal data
            test_entities = {"age": 30}
            test_chunks = [{
                "text": "Test policy clause about grace periods.",
                "metadata": {"file_name": "test.pdf"},
                "similarity_score": 0.8
            }]
            
            analysis = self.llm_analyzer.analyze_query(
                "Test query", test_entities, test_chunks
            )
            
            if analysis.decision in ["APPROVED", "REJECTED", "REQUIRES_REVIEW"]:
                return {
                    "status": "passed", 
                    "message": f"LLM analysis successful - Decision: {analysis.decision}"
                }
            else:
                return {"status": "failed", "error": f"Invalid decision: {analysis.decision}"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _test_end_to_end(self, test_query: str) -> Dict[str, Any]:
        """Test complete end-to-end pipeline"""
        try:
            response = self.process_query(test_query)
            
            required_fields = ['query', 'decision', 'justification']
            missing_fields = [f for f in required_fields if not getattr(response, f, None)]
            
            if missing_fields:
                return {"status": "failed", "error": f"Missing response fields: {missing_fields}"}
            
            return {
                "status": "passed", 
                "message": f"End-to-end test successful - Decision: {response.decision}",
                "processing_time": f"{response.processing_time:.2f}s"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


def main():
    """Main function to run the complete system with Ollama"""
    print("🚀 LLM Document Processing System (Open Source)")
    print("=" * 60)
    print("🤖 Using Ollama for local LLM processing")
    print("💡 No API keys required - completely free!")
    print()
    
    try:
        # Initialize system
        processor = LLMDocumentProcessor()
        
        # Setup system
        print("⚙️ Setting up system...")
        if not processor.setup_system():
            print("❌ System setup failed!")
            print("💡 Make sure Ollama is running: ollama serve")
            print("💡 Download a model: ollama pull llama3")
            return
        
        # Run tests
        print("\n🧪 Running system tests...")
        test_results = processor.test_system()
        
        if test_results["overall_status"] == "passed":
            print("✅ All tests passed!")
        else:
            print(f"⚠️ Tests status: {test_results['overall_status']}")
            
            # Show specific failures
            for test_name, result in test_results.get("tests", {}).items():
                if result.get("status") != "passed":
                    print(f"  ❌ {test_name}: {result.get('error', 'Failed')}")
        
        # Show system stats
        print("\n📊 System Statistics:")
        stats = processor.get_system_stats()
        print(f"  LLM Provider: {stats.get('llm_provider', 'unknown')}")
        
        for component, info in stats.get("components", {}).items():
            status = info.get("status", "unknown")
            if component == "llm_analyzer":
                model = info.get("model", "unknown")
                print(f"  {component}: {status} ({model})")
            else:
                print(f"  {component}: {status}")
        
        # Interactive mode
        print("\n" + "=" * 60)
        print("🔍 Interactive Query Mode")
        print("Type 'quit' to exit, 'stats' for system info, 'models' for Ollama models")
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query.lower() == 'stats':
                    stats = processor.get_system_stats()
                    print(json.dumps(stats, indent=2))
                    continue
                
                if query.lower() == 'models':
                    model_info = processor.llm_analyzer.get_model_info()
                    print(f"Current model: {model_info.get('current_model', 'unknown')}")
                    print(f"Available models: {', '.join(model_info.get('available_models', []))}")
                    continue
                
                if not query:
                    continue
                
                # Process query
                print("\n🤖 Processing with Ollama...")
                response = processor.process_query(query)
                
                # Display results
                print(f"\n📋 Query: {response.query}")
                print(f"✅ Decision: {response.decision.upper()}")
                print(f"💰 Amount: {response.amount}")
                print(f"🎯 Confidence: {response.confidence:.2f}")
                print(f"⏱️ Processing Time: {response.processing_time:.2f}s")
                print(f"🤖 LLM Provider: {response.llm_provider}")
                print(f"📝 Justification: {response.justification[:300]}...")
                
                if response.relevant_clauses:
                    print(f"📄 Referenced {len(response.relevant_clauses)} policy clauses")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("💡 Make sure Ollama is running: ollama serve")
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"❌ Failed to initialize system: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Install Ollama: https://ollama.com/")
        print("  2. Start Ollama: ollama serve")
        print("  3. Download model: ollama pull llama3")
        print("  4. Check configuration files exist")


if __name__ == "__main__":
    main()