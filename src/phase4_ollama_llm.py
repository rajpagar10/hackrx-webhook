"""
Phase 4: LLM Analysis Module (Open Source - Ollama Integration)
Handles local LLM integration using Ollama for insurance policy analysis
"""
import json
import re
import os
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
import yaml
from datetime import datetime

@dataclass
class PolicyAnalysis:
    """Data structure for policy analysis results"""
    decision: str  # "approved", "rejected", "requires_review"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    relevant_clauses: List[Dict[str, Any]]
    entities: Dict[str, Any]
    amount: Optional[float] = None
    conditions: Optional[List[str]] = None

class OllamaLLMAnalyzer:
    """
    Ollama-powered LLM analyzer for insurance policy analysis
    Uses local models like Llama 3, Mistral, Gemma 2, etc.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Ollama LLM analyzer"""
        self.config = self._load_config(config_path)
        
        # Ollama settings from config and environment
        self.model = os.getenv('LLM_MODEL', self.config['llm']['model'])
        self.base_url = os.getenv('OLLAMA_API_BASE', self.config['llm']['base_url'])
        self.temperature = self.config['llm'].get('temperature', 0.1)
        self.max_tokens = self.config['llm'].get('max_tokens', 2000)
        self.timeout = self.config['llm'].get('timeout', 180)
        
        # Insurance-specific system prompt
        self.system_prompt = self._create_insurance_system_prompt()
        
        # Verify Ollama setup
        self._verify_ollama_setup()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            # Return minimal default config
            return {
                'llm': {
                    'provider': 'ollama',
                    'model': 'llama3',
                    'base_url': 'http://localhost:11434',
                    'temperature': 0.1,
                    'max_tokens': 2000,
                    'timeout': 60
                }
            }

    def _verify_ollama_setup(self):
        """Verify Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                
                if self.model in model_names:
                    logger.success(f"✅ Ollama ready - Model: {self.model}")
                else:
                    logger.warning(f"⚠️ Model '{self.model}' not found")
                    logger.info(f"Available models: {model_names}")
                    logger.info(f"To download: ollama pull {self.model}")
            else:
                logger.error(f"❌ Ollama API returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.info("💡 Make sure Ollama is running:")
            logger.info("   1. Install: https://ollama.com/")
            logger.info("   2. Start: ollama serve")
            logger.info(f"   3. Download model: ollama pull {self.model}")

    def _create_insurance_system_prompt(self) -> str:
        """Create specialized system prompt for insurance policy analysis"""
        return """You are an expert insurance policy analyst AI specialized in Indian insurance policies. Your task is to analyze insurance queries against policy documents and make accurate decisions.

CORE RESPONSIBILITIES:
1. Analyze user queries with extracted entities (age, procedure, location, policy duration)
2. Review provided policy document chunks for relevant clauses  
3. Apply insurance rules to determine coverage eligibility
4. Make clear decisions: APPROVED, REJECTED, or REQUIRES_REVIEW
5. Provide detailed reasoning with specific policy clause references
6. Calculate claim amounts when determinable from policy terms

DECISION FRAMEWORK:
- APPROVED: Query clearly matches covered benefits with all eligibility criteria met
- REJECTED: Query explicitly excluded, waiting periods not met, or clearly not covered
- REQUIRES_REVIEW: Ambiguous cases, insufficient information, or edge cases requiring human judgment

INSURANCE ANALYSIS RULES:
- Always check waiting periods (initial, specific disease, pre-existing conditions)
- Verify age eligibility and policy duration requirements
- Consider sub-limits, co-payments, and deductibles
- Check for specific exclusions related to the query
- Factor in geographic restrictions if mentioned
- Validate procedure codes and medical necessity

RESPONSE FORMAT:
You MUST respond in valid JSON format with these exact fields:
{
  "decision": "APPROVED|REJECTED|REQUIRES_REVIEW",
  "confidence": 0.85,
  "reasoning": "Detailed explanation referencing specific policy clauses and analysis logic...",
  "amount": 50000.0,
  "conditions": ["Pre-authorization required", "Network hospital only"],
  "relevant_clauses": [
    {
      "clause_text": "Exact text from policy document...",
      "source_document": "policy_document.pdf", 
      "section": "Section 3.2.1",
      "relevance": "Defines waiting period for orthopedic procedures"
    }
  ]
}

IMPORTANT: 
- Use conservative analysis - when uncertain, choose REQUIRES_REVIEW
- Reference specific policy sections and clause numbers when available
- Consider Indian insurance regulations and common policy structures
- Be precise with amounts and conditions"""

    def analyze_query(
        self, 
        query: str, 
        entities: Dict[str, Any], 
        retrieved_chunks: List[Dict[str, Any]]
    ) -> PolicyAnalysis:
        """
        Analyze insurance query using Ollama LLM
        
        Args:
            query: User's original query
            entities: Extracted entities (age, procedure, location, etc.)
            retrieved_chunks: Relevant document chunks from semantic search
            
        Returns:
            PolicyAnalysis: Structured analysis result
        """
        logger.info(f"🤖 Analyzing with Ollama ({self.model}): '{query[:80]}...'")
        
        # Prepare analysis context
        context = self._prepare_insurance_context(query, entities, retrieved_chunks)
        
        try:
            # Call Ollama API
            response_text = self._call_ollama_api(context)
            
            # Parse JSON response
            analysis_data = self._parse_json_response(response_text)
            
            # Create PolicyAnalysis object
            analysis = PolicyAnalysis(
                decision=analysis_data.get('decision', 'REQUIRES_REVIEW'),
                confidence=float(analysis_data.get('confidence', 0.5)),
                reasoning=analysis_data.get('reasoning', 'Analysis completed'),
                relevant_clauses=analysis_data.get('relevant_clauses', []),
                entities=entities,
                amount=analysis_data.get('amount'),
                conditions=analysis_data.get('conditions', [])
            )
            
            # Validate and enhance analysis
            validated_analysis = self._validate_analysis(analysis)
            
            logger.success(f"✅ Analysis complete - {validated_analysis.decision} (confidence: {validated_analysis.confidence:.2f})")
            return validated_analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._create_error_analysis(query, entities, str(e))

    def _call_ollama_api(self, context: str) -> str:
        """Call Ollama API with the prepared context"""
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": context
                }
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_k": 40,
                "top_p": 0.9
            },
            "stream": False
        }
        
        logger.debug(f"Calling Ollama API: {self.base_url}/api/chat")
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['message']['content']
        else:
            error_msg = f"Ollama API error: HTTP {response.status_code}"
            try:
                error_detail = response.json().get('error', response.text)
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {response.text}"
            
            raise Exception(error_msg)

    def _prepare_insurance_context(
        self, 
        query: str, 
        entities: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> str:
        """Prepare comprehensive context for insurance analysis"""
        
        context_parts = [
            "=== INSURANCE POLICY ANALYSIS REQUEST ===",
            "",
            f"USER QUERY: {query}",
            "",
            "EXTRACTED ENTITIES:"
        ]
        
        # Add extracted entities
        if entities:
            for key, value in entities.items():
                context_parts.append(f"• {key.title()}: {value}")
        else:
            context_parts.append("• No specific entities extracted from query")
        
        context_parts.extend([
            "",
            "RELEVANT POLICY DOCUMENT SECTIONS:",
            "=" * 50
        ])
        
        # Add retrieved chunks with enhanced metadata
        for i, chunk in enumerate(chunks[:5], 1):  # Limit to top 5 for context management
            metadata = chunk.get('metadata', {})
            similarity = chunk.get('similarity_score', 0.0)
            
            context_parts.extend([
                f"",
                f"DOCUMENT SECTION {i} (Relevance Score: {similarity:.3f})",
                f"Source: {metadata.get('file_name', 'Unknown Document')}",
                f"Chunk ID: {metadata.get('chunk_id', 'N/A')}",
                f"Document Type: {metadata.get('doc_type', 'Unknown')}",
                f"",
                f"Content:",
                f"{chunk.get('text', 'No content available')[:1200]}...",  # Limit chunk length
            ])
        
        context_parts.extend([
            "",
            "=== ANALYSIS INSTRUCTIONS ===",
            "Please analyze this insurance query against the provided policy documents.",
            "Consider all relevant factors including:",
            "- Waiting periods and policy age",
            "- Coverage limits and exclusions", 
            "- Eligibility criteria",
            "- Geographic and network restrictions",
            "- Pre-existing condition clauses",
            "",
            "Provide your analysis in the specified JSON format with detailed reasoning."
        ])
        
        return "\n".join(context_parts)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON with robust error handling"""
        
        # Clean response text
        cleaned_text = response_text.strip()
        
        try:
            # Try direct JSON parsing first
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            json_patterns = [
                r'```json\s*({\s*.*?\s*})\s*```',  # JSON code blocks
                r'```\s*({\s*.*?\s*})\s*```',      # Generic code blocks
                r'(\{\s*["\']decision["\'].*?\})',   # Find decision object
                r'(\{.*?\})'                       # Any JSON-like object
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: Extract structured information from text
            logger.warning("Could not parse JSON, extracting structured info from text")
            return self._extract_structured_data_from_text(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._extract_structured_data_from_text(response_text)

    def _extract_structured_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured information when JSON parsing fails"""
        
        result = {
            "decision": "REQUIRES_REVIEW",
            "confidence": 0.3,
            "reasoning": "Could not parse structured response from LLM. Manual review recommended.",
            "relevant_clauses": []
        }
        
        text_lower = text.lower()
        
        # Extract decision
        if any(word in text_lower for word in ["approved", "approve", "covered", "eligible"]):
            result["decision"] = "APPROVED"
        elif any(word in text_lower for word in ["rejected", "reject", "denied", "not covered", "excluded"]):
            result["decision"] = "REJECTED"
        elif any(word in text_lower for word in ["review", "manual", "unclear", "uncertain"]):
            result["decision"] = "REQUIRES_REVIEW"
        
        # Extract confidence if mentioned
        conf_match = re.search(r"confidence[:\s]*([0-9.]+)", text_lower)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
                if result["confidence"] > 1.0:  # Handle percentage format
                    result["confidence"] = result["confidence"] / 100.0
            except ValueError:
                pass
        
        # Extract amount (Indian currency formats)
        amount_patterns = [
            r"(?:rs\.?\s*|₹\s*)([0-9,]+(?:\.[0-9]{2})?)",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*(?:rs\.?|rupees?)",
            r"amount[:\s]*(?:rs\.?\s*|₹\s*)?([0-9,]+(?:\.[0-9]{2})?)"
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    result["amount"] = float(amount_str)
                    break
                except ValueError:
                    continue
        
        # Use original text as reasoning (truncated)
        if len(text) > 500:
            result["reasoning"] = text[:500] + "... [Response truncated due to parsing issues]"
        else:
            result["reasoning"] = text
        
        return result

    def _validate_analysis(self, analysis: PolicyAnalysis) -> PolicyAnalysis:
        """Validate and enhance the analysis result"""
        
        # Ensure decision is valid
        valid_decisions = ["APPROVED", "REJECTED", "REQUIRES_REVIEW"]
        if analysis.decision not in valid_decisions:
            logger.warning(f"Invalid decision '{analysis.decision}', defaulting to REQUIRES_REVIEW")
            analysis.decision = "REQUIRES_REVIEW"
            analysis.confidence = min(analysis.confidence, 0.5)
        
        # Ensure confidence is in valid range
        if not (0.0 <= analysis.confidence <= 1.0):
            logger.warning(f"Invalid confidence {analysis.confidence}, clamping to valid range")
            analysis.confidence = max(0.0, min(1.0, analysis.confidence))
        
        # Add validation conditions
        if not analysis.conditions:
            analysis.conditions = []
        
        # Add conservative conditions based on entities
        if analysis.entities.get('age'):
            age = analysis.entities['age']
            if age < 18:
                analysis.conditions.append("Minor - verify guardian consent and eligibility")
            elif age > 80:
                analysis.conditions.append("Senior citizen - verify age-based coverage limits")
        
        if analysis.entities.get('policy_duration'):
            duration = str(analysis.entities['policy_duration']).lower()
            if 'month' in duration and any(char.isdigit() for char in duration):
                months = int(''.join(filter(str.isdigit, duration)))
                if months < 12:
                    analysis.conditions.append(f"New policy ({months} months) - limited coverage may apply")
        
        # High amount verification
        if analysis.amount and analysis.amount > 100000:  # 1 lakh
            analysis.conditions.append("High value claim - requires additional verification")
        
        return analysis

    def _create_error_analysis(self, query: str, entities: Dict[str, Any], error: str) -> PolicyAnalysis:
        """Create fallback analysis when LLM processing fails"""
        return PolicyAnalysis(
            decision="REQUIRES_REVIEW",
            confidence=0.0,
            reasoning=f"Automated analysis failed due to: {error}. This query requires manual review by an insurance expert.",
            relevant_clauses=[],
            entities=entities,
            conditions=[
                "System error occurred during analysis",
                "Manual review required",
                "Escalate to insurance specialist"
            ]
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Ollama setup"""
        try:
            # Get model information
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                current_model = next((m for m in models if m['name'].startswith(self.model)), None)
                
                return {
                    "provider": "ollama",
                    "base_url": self.base_url,
                    "current_model": self.model,
                    "model_info": current_model,
                    "available_models": [m['name'] for m in models],
                    "status": "connected"
                }
        except Exception as e:
            return {
                "provider": "ollama",
                "base_url": self.base_url,
                "current_model": self.model,
                "status": "disconnected",
                "error": str(e)
            }

    def test_connection(self) -> bool:
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    """Test the Ollama LLM analyzer"""
    
    print("🤖 Testing Ollama LLM Analysis Module\n")
    print("=" * 60)
    
    # Initialize analyzer
    try:
        analyzer = OllamaLLMAnalyzer()
        
        # Show model info
        info = analyzer.get_model_info()
        print(f"🔧 Provider: {info['provider']}")
        print(f"🏠 Base URL: {info['base_url']}")
        print(f"🤖 Model: {info['current_model']}")
        print(f"📊 Status: {info['status']}")
        
        if info['status'] == 'disconnected':
            print(f"❌ Error: {info.get('error', 'Unknown error')}")
            print("\n💡 Troubleshooting:")
            print("   1. Make sure Ollama is installed: https://ollama.com/")
            print("   2. Start Ollama: ollama serve")
            print(f"   3. Download model: ollama pull {analyzer.model}")
            return
        
        print(f"✅ Available models: {', '.join(info['available_models'][:3])}...")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "query": "What is the grace period for premium payment?",
            "entities": {},
            "retrieved_chunks": [
                {
                    "text": "Grace period for premium payment is 30 days from the due date. During grace period, coverage continues without interruption. No penalty or additional charges apply during grace period.",
                    "metadata": {"file_name": "policy_document.pdf", "chunk_id": "grace-period-1"},
                    "similarity_score": 0.95
                }
            ]
        },
        {
            "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "entities": {
                "age": 46, 
                "gender": "male", 
                "procedure": "knee surgery", 
                "location": "Pune", 
                "policy_duration": "3 months"
            },
            "retrieved_chunks": [
                {
                    "text": "Knee surgery and joint replacement procedures are covered under this policy after completion of 24 months waiting period from policy inception date. Coverage includes pre-operative, operative and post-operative expenses.",
                    "metadata": {"file_name": "policy_terms.pdf", "chunk_id": "orthopedic-coverage-1"},
                    "similarity_score": 0.89
                },
                {
                    "text": "For policies less than 12 months old, only accidental injuries are covered for surgical procedures. Pre-planned surgeries require policy to be active for minimum 2 years.",
                    "metadata": {"file_name": "waiting_periods.pdf", "chunk_id": "new-policy-terms-1"},  
                    "similarity_score": 0.82
                }
            ]
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}: {test_case['query']}")
        print("-" * 40)
        
        try:
            # Analyze query
            analysis = analyzer.analyze_query(
                query=test_case['query'],
                entities=test_case['entities'],
                retrieved_chunks=test_case['retrieved_chunks']
            )
            
            # Display results
            print(f"✅ Decision: {analysis.decision}")
            print(f"🎯 Confidence: {analysis.confidence:.2f}")
            print(f"💰 Amount: {analysis.amount}")
            print(f"📋 Conditions: {len(analysis.conditions or [])}")
            if analysis.conditions:
                for condition in analysis.conditions[:2]:  # Show first 2 conditions
                    print(f"   • {condition}")
            print(f"🧠 Reasoning: {analysis.reasoning[:200]}...")
            print(f"📄 Referenced Clauses: {len(analysis.relevant_clauses)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ Ollama LLM analysis tests completed!")
    
    # Connection test
    if analyzer.test_connection():
        print("🔗 Connection Status: ✅ Connected")
    else:
        print("🔗 Connection Status: ❌ Disconnected")


if __name__ == "__main__":
    main()