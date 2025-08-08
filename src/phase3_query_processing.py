"""
Phase 3: Query Processing Module (FIXED VERSION)
Handles query normalization, classification, and canonicalization
"""
import re
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from loguru import logger
import yaml


class QueryProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize query processor with configuration"""
        self.config = self._load_config(config_path)
        
        # Load embedding model for query encoding
        model_name = self.config['embeddings']['model_name']
        self.model = SentenceTransformer(model_name)
        
        # Query processing settings
        self.preserve_abbreviations = self.config['query_processing']['normalization']['preserve_abbreviations']
        self.categories = self.config['query_processing']['categories']
        
        # Initialize category keywords and synonym mappings
        self.category_keywords = self._initialize_category_keywords()
        self.synonym_map = self._initialize_synonym_map()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            # Return minimal default config
            return {
                'embeddings': {
                    'model_name': 'BAAI/bge-base-en-v1.5',
                    'instruction_prefix': 'Represent this sentence for searching relevant passages: ',
                    'normalize': True
                },
                'query_processing': {
                    'normalization': {'preserve_abbreviations': ['PED', 'NCD', 'ICU', 'AYUSH']},
                    'categories': ['Coverage', 'Waiting Period', 'Eligibility', 'Limits', 'Definition', 'Discounts', 'Benefits', 'Hospitalization', 'AYUSH', 'Maternity']
                }
            }

    def _initialize_category_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for query classification"""
        return {
            "Coverage": [
                "cover", "coverage", "included", "include", "insured", "covered", 
                "benefit", "protection", "scope", "what is covered"
            ],
            "Waiting Period": [
                "waiting period", "how long", "after how many", "time before", 
                "initial waiting", "when can I claim", "eligibility period"
            ],
            "Eligibility": [
                "eligibility", "who is eligible", "criteria", "conditions to apply",
                "qualification", "requirements", "who can apply"
            ],
            "Limits": [
                "limit", "maximum", "cap", "restricted", "sub-limit", "restriction",
                "ceiling", "upper limit", "maximum amount"
            ],
            "Definition": [
                "define", "definition", "what is", "meaning", "explain", "what does mean"
            ],
            "Discounts": [
                "no claim", "ncd", "discount", "bonus", "reduction", "cashback",
                "rebate", "saving"
            ],
            "Benefits": [
                "benefit", "advantage", "reward", "perk", "preventive", 
                "additional benefit", "extra coverage"
            ],
            "Hospitalization": [
                "hospital", "icu", "room rent", "admission", "hospitalization",
                "inpatient", "daycare", "nursing"
            ],
            "AYUSH": [
                "ayurveda", "homeopathy", "ayush", "unani", "naturopathy", 
                "siddha", "alternative medicine"
            ],
            "Maternity": [
                "maternity", "childbirth", "pregnancy", "delivery", "termination",
                "newborn", "pre-natal", "post-natal"
            ]
        }

    def _initialize_synonym_map(self) -> Dict[str, str]:
        """Initialize synonym mappings for canonicalization"""
        return {
            # Eye-related terms
            "eye operation": "cataract surgery",
            "eye surgery": "cataract surgery",
            "vision surgery": "cataract surgery",
            
            # Claim-related terms
            "no claim bonus": "no claim discount",
            "bonus": "no claim discount",
            "ncb": "no claim discount",
            "ncd": "no claim discount",
            
            # Medical terms
            "checkup": "health check-up",
            "preventive checkup": "preventive health check-up",
            "routine checkup": "preventive health check-up",
            
            # Pregnancy/maternity
            "pregnancy cost": "maternity expenses",
            "delivery cost": "maternity expenses",
            "childbirth cost": "maternity expenses",
            
            # Hospital terms
            "hospital definition": "hospital",
            "room charges": "room rent",
            "icu charges": "icu charges",
            
            # Insurance terms
            "ped": "pre-existing disease",
            "pre existing": "pre-existing disease",
            "waiting time": "waiting period",
            "grace time": "grace period",
            
            # Common abbreviations
            "o.p.d": "outpatient department",
            "opd": "outpatient department",
            "i.p.d": "inpatient department", 
            "ipd": "inpatient department"
        }

    def normalize_query(self, query: str) -> str:
        """
        Normalize user query by cleaning and standardizing text
        FIXED VERSION - prevents over-aggressive text removal
        
        Args:
            query (str): Raw user input
            
        Returns:
            str: Normalized query
        """
        logger.debug(f"Normalizing query: {query}")
        
        # Convert to lowercase
        normalized = query.lower()
        
        # FIXED: More conservative punctuation removal
        # Only remove specific punctuation marks, keep letters/numbers/spaces
        normalized = re.sub(r"[?!.,;:\"\']", " ", normalized)  # Replace with spaces
        
        # Keep alphanumeric characters, spaces, hyphens, and common symbols
        normalized = re.sub(r"[^a-zA-Z0-9\s\-/()]", "", normalized)
        
        # Collapse multiple whitespaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        logger.debug(f"Normalized to: {normalized}")
        
        # VALIDATION: Check if normalization didn't destroy the query
        if len(normalized) < 3:
            logger.warning(f"Normalization resulted in very short text: '{normalized}'. Using original query.")
            # Fallback: minimal normalization
            normalized = query.lower().strip()
            normalized = re.sub(r"[?!.,;:\"\']", " ", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized

    def classify_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Classify query into predefined categories
        
        Args:
            query (str): Normalized query
            
        Returns:
            Tuple[str, List[str]]: (category, matched_keywords)
        """
        matched_keywords = []
        matched_category = "Unknown"
        max_matches = 0
        
        # Check each category for keyword matches
        for category, keywords in self.category_keywords.items():
            category_matches = []
            for keyword in keywords:
                # Use word boundary regex to match complete phrases
                pattern = rf"\\b{re.escape(keyword)}\\b"
                if re.search(pattern, query, re.IGNORECASE):
                    category_matches.append(keyword)
                    logger.debug(f"Found keyword '{keyword}' in category '{category}'")
            
            # Choose category with most matches
            if len(category_matches) > max_matches:
                max_matches = len(category_matches)
                matched_category = category
                matched_keywords = category_matches
        
        # If no matches, try partial matching
        if matched_category == "Unknown":
            for category, keywords in self.category_keywords.items():
                for keyword in keywords:
                    if keyword in query:
                        matched_keywords.append(keyword)
                        matched_category = category
                        logger.debug(f"Partial match: '{keyword}' in category '{category}'")
                        break
                if matched_category != "Unknown":
                    break
        
        return matched_category, matched_keywords

    def canonicalize_query(self, query: str) -> str:
        """
        Replace synonyms and variations with canonical terms
        
        Args:
            query (str): Normalized query
            
        Returns:
            str: Canonicalized query
        """
        canonicalized = query
        
        # Apply synonym mappings (case insensitive)
        for synonym, canonical in self.synonym_map.items():
            pattern = rf"\\b{re.escape(synonym)}\\b"
            canonicalized = re.sub(pattern, canonical, canonicalized, flags=re.IGNORECASE)
        
        logger.debug(f"Canonicalized '{query}' to '{canonicalized}'")
        return canonicalized

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract key entities from the query (age, procedure, location, etc.)
        
        Args:
            query (str): Original query
            
        Returns:
            Dict[str, Any]: Extracted entities
        """
        entities = {}
        
        # Extract age patterns (more robust)
        age_patterns = [
            r"(\\d{1,3})\\s*(?:year|yr|y)(?:s)?(?:\\s*old)?",  # "46 years old", "25 yr"
            r"(\\d{1,3})\\s*(?:-|\\s)?(?:year|yr)(?:s)?\\s*old",  # "46-year-old", "25 year old"
            r"age\\s*(\\d{1,3})",  # "age 46"
            r"(\\d{1,3})M|F",  # "46M", "25F"
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and match.group(1):
                try:
                    age = int(match.group(1))
                    if 0 <= age <= 120:  # Reasonable age range
                        entities["age"] = age
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract gender
        gender_patterns = [
            r"\\b(male|M)\\b",
            r"\\b(female|F)\\b"
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                gender_text = match.group(1).lower()
                if gender_text in ['male', 'm']:
                    entities["gender"] = "male"
                elif gender_text in ['female', 'f']:
                    entities["gender"] = "female"
                break
        
        # Extract common procedures (expanded list)
        procedures = [
            "knee surgery", "cataract surgery", "heart surgery", "dental treatment",
            "physiotherapy", "chemotherapy", "dialysis", "bypass surgery",
            "eye surgery", "eye operation", "joint replacement", "appendectomy",
            "gallbladder surgery", "hernia surgery", "plastic surgery"
        ]
        
        for procedure in procedures:
            if re.search(rf"\\b{re.escape(procedure)}\\b", query, re.IGNORECASE):
                entities["procedure"] = procedure
                break
        
        # Extract Indian cities/locations (expanded)
        locations = [
            "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata",
            "pune", "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur",
            "indore", "bhopal", "visakhapatnam", "patna", "surat", "agra",
            "meerut", "rajkot", "kalyan", "vasai", "aurangabad", "dhanbad"
        ]
        
        for location in locations:
            if re.search(rf"\\b{re.escape(location)}\\b", query, re.IGNORECASE):
                entities["location"] = location.title()
                break
        
        # Extract policy duration (improved patterns)
        duration_patterns = [
            r"(\\d+)\\s*(?:month|months|mon)(?:\\s*(?:old|policy))?",
            r"(\\d+)\\s*(?:year|years|yr)(?:\\s*(?:old|policy))?",
            r"(\\d+)\\s*(?:M|Y)\\s*(?:old|policy)"
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["policy_duration"] = match.group(0).strip()
                break
        
        # Extract amounts (Indian currency format)
        amount_patterns = [
            r"(?:rs\\.?|rupees?)\\s*(\\d+(?:,\\d+)*(?:\\.\\d{2})?)",  # Rs. 50,000
            r"(\\d+(?:,\\d+)*(?:\\.\\d{2})?)\\s*(?:rs\\.?|rupees?)",  # 50,000 Rs
            r"₹\\s*(\\d+(?:,\\d+)*(?:\\.\\d{2})?)",  # ₹50,000
            r"inr\\s*(\\d+(?:,\\d+)*(?:\\.\\d{2})?)",  # INR 50000
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    entities["amount"] = float(amount_str)
                    break
                except ValueError:
                    continue
        
        if entities:
            logger.info(f"Extracted entities: {entities}")
        
        return entities

    def get_query_embedding(self, canonical_query: str) -> np.ndarray:
        """
        Convert canonicalized query to embedding vector
        
        Args:
            canonical_query (str): Preprocessed query
            
        Returns:
            np.ndarray: Query embedding vector
        """
        try:
            # Add instruction prefix for better retrieval
            instruction = self.config['embeddings']['instruction_prefix']
            query_with_instruction = instruction + canonical_query
            
            # Generate embedding
            embedding = self.model.encode(
                query_with_instruction,
                normalize_embeddings=self.config['embeddings']['normalize']
            )
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # BGE model default dimension

    def process_query(self, raw_query: str) -> Dict[str, Any]:
        """
        Complete query processing pipeline
        
        Args:
            raw_query (str): Raw user input
            
        Returns:
            Dict[str, Any]: Processed query information
        """
        logger.info(f"Processing query: '{raw_query}'")
        
        try:
            # Step 1: Normalize query
            normalized = self.normalize_query(raw_query)
            
            # Step 2: Extract entities from original query (before normalization)
            entities = self.extract_entities(raw_query)
            
            # Step 3: Classify query
            category, keywords = self.classify_query(normalized)
            
            # Step 4: Canonicalize query
            canonical = self.canonicalize_query(normalized)
            
            # Step 5: Generate embedding
            embedding = self.get_query_embedding(canonical)
            
            result = {
                "original_query": raw_query,
                "normalized_query": normalized,
                "canonical_query": canonical,
                "category": category,
                "matched_keywords": keywords,
                "entities": entities,
                "embedding": embedding,
                "embedding_shape": embedding.shape
            }
            
            logger.info(f"Query processed - Category: {category}, Entities: {len(entities)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "original_query": raw_query,
                "normalized_query": raw_query.lower(),
                "canonical_query": raw_query.lower(),
                "category": "Unknown",
                "matched_keywords": [],
                "entities": {},
                "embedding": np.zeros(384),
                "embedding_shape": (384,),
                "error": str(e)
            }

    def batch_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            queries (List[str]): List of raw queries
            
        Returns:
            List[Dict[str, Any]]: List of processed query results
        """
        logger.info(f"Processing {len(queries)} queries in batch...")
        
        results = []
        for query in queries:
            try:
                result = self.process_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    "original_query": query,
                    "error": str(e)
                })
        
        return results


def main():
    """Main function to test query processing"""
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "Does the plan include eye operation?",
        "Is there any bonus if I don't make a claim?", 
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "What is the waiting period for cataract surgery?",
        "How to file a cashless claim for ₹50,000?",
        "Are maternity expenses covered under this policy?"
    ]
    
    print("🔍 Testing Query Processing Pipeline (FIXED VERSION)\\n")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n📝 Test Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = processor.process_query(query)
            
            print(f"✅ Original:   {result['original_query']}")
            print(f"🔄 Normalized: {result['normalized_query']}")
            print(f"📝 Canonical:  {result['canonical_query']}")
            print(f"🏷️  Category:   {result['category']}")
            print(f"🔑 Keywords:   {result['matched_keywords']}")
            print(f"📊 Entities:   {result['entities']}")
            print(f"🧮 Embedding:  Shape {result['embedding_shape']}")
            
            # Validate normalization didn't break
            if len(result['normalized_query']) < 5:
                print("⚠️  WARNING: Normalization may be too aggressive!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\\n" + "="*70)
    print("✅ Query processing tests completed!")


if __name__ == "__main__":
    main()