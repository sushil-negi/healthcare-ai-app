"""
Healthcare AI Engine with Advanced Response Generation
Uses 525K training conversations for context-aware, unique responses
"""

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# For proper LLM integration (optional)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM libraries not available. Running in knowledge-base only mode.")


class HealthcareAIEngine:
    """Advanced Healthcare AI with both LLM and knowledge-based responses"""

    def __init__(self, use_llm=True, model_name="microsoft/DialoGPT-small"):
        if not LLM_AVAILABLE:
            use_llm = False
        self.use_llm = use_llm
        self.conversation_history = []
        self.knowledge_base = self._load_knowledge_base()
        self.response_cache = {}
        self.llm_pipeline = None

        if use_llm:
            try:
                # Initialize language model for unique response generation
                logger.info(f"Loading LLM: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)

                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Create pipeline for text generation
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=150,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                )
                logger.info("LLM loaded successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to load LLM: {e}. Falling back to knowledge base only."
                )
                self.use_llm = False

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load comprehensive healthcare knowledge from training data"""
        knowledge: Dict[str, Any] = {
            "conversations": defaultdict(list),
            "responses_by_category": defaultdict(list),
            "crisis_responses": [],
            "safety_disclaimers": [],
            "total_conversations": 525017,
        }

        # Load sample conversations from JSONL file
        try:
            with open(
                "/Users/snegi/Documents/github/mlops-project/data/combined_healthcare_training_data.jsonl",
                "r",
            ) as f:
                # Skip header lines
                next(f)
                next(f)

                # Load up to 10000 conversations for memory efficiency
                for i, line in enumerate(f):
                    if i >= 10000:
                        break

                    try:
                        conv = json.loads(line.strip())
                        if "messages" in conv and len(conv["messages"]) >= 2:
                            category = conv.get("category", "general")
                            user_msg = conv["messages"][0]["content"]
                            assistant_msg = conv["messages"][1]["content"]

                            # Store conversations by category
                            knowledge["conversations"][category].append(
                                {
                                    "user": user_msg,
                                    "assistant": assistant_msg,
                                    "metadata": conv.get("metadata", {}),
                                }
                            )

                            # Extract response patterns
                            knowledge["responses_by_category"][category].append(
                                assistant_msg
                            )

                            # Collect crisis responses
                            if any(
                                word in user_msg.lower()
                                for word in ["suicide", "crisis", "emergency"]
                            ):
                                knowledge["crisis_responses"].append(assistant_msg)
                    except (KeyError, TypeError, ValueError):
                        continue

            conversations_dict = knowledge["conversations"]
            total_conversations = sum(len(v) for v in conversations_dict.values())
            logger.info(f"Loaded {total_conversations} conversations")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")

        # Add comprehensive category mappings
        knowledge["category_keywords"] = {
            "adl": [
                "mobility",
                "walking",
                "transfer",
                "wheelchair",
                "balance",
                "falling",
                "bathing",
                "dressing",
                "eating",
                "toileting",
                "daily activities",
                "independence",
                "adaptive",
            ],
            "senior_care": [
                "elderly",
                "senior",
                "aging",
                "parent",
                "grandfather",
                "grandmother",
                "lonely",
                "isolation",
                "memory",
                "medication",
                "aging in place",
                "geriatric",
            ],
            "mental_health": [
                "anxiety",
                "depression",
                "stress",
                "panic",
                "worried",
                "sad",
                "overwhelmed",
                "mental health",
                "therapy",
                "counseling",
                "emotional",
                "mood",
            ],
            "respite_care": [
                "caregiver",
                "exhausted",
                "break",
                "respite",
                "burnout",
                "caring for",
                "relief",
                "temporary care",
                "support",
                "overwhelmed",
            ],
            "disabilities": [
                "disability",
                "wheelchair",
                "adaptive",
                "accessibility",
                "accommodation",
                "assistive",
                "impairment",
                "special needs",
                "inclusion",
            ],
        }

        return knowledge

    def _detect_category(self, text: str) -> str:
        """Detect conversation category based on keywords"""
        text_lower = text.lower()

        # Check for crisis first
        crisis_words = [
            "suicide",
            "kill myself",
            "hurt myself",
            "want to die",
            "end it all",
        ]
        if any(word in text_lower for word in crisis_words):
            return "crisis"

        # Score each category
        category_scores = {}
        for category, keywords in self.knowledge_base["category_keywords"].items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score

        # Return highest scoring category or 'general'
        if category_scores:
            return max(category_scores, key=lambda x: category_scores[x])
        return "general"

    def _get_similar_conversation(
        self, user_input: str, category: str
    ) -> Optional[Dict]:
        """Find similar conversation from knowledge base"""
        # First try the detected category
        conversations = self.knowledge_base["conversations"].get(category, [])

        # If no conversations in category, search all categories
        if not conversations:
            all_conversations = []
            for cat_convs in self.knowledge_base["conversations"].values():
                all_conversations.extend(cat_convs)
            conversations = all_conversations

        if not conversations:
            return None

        # Simple similarity scoring based on common words
        user_words = set(user_input.lower().split())
        best_match = None
        best_score = 0

        # Check more conversations for better matches
        for conv in conversations[:500]:  # Check first 500 for better coverage
            conv_words = set(conv["user"].lower().split())
            score = len(user_words.intersection(conv_words))
            if score > best_score:
                best_score = score
                best_match = conv

        # Lower threshold for match to increase response variety
        return best_match if best_score > 1 else None

    def _generate_llm_response(
        self, user_input: str, context: str = ""
    ) -> Optional[str]:
        """Generate response using language model"""
        if not self.llm_pipeline:
            return None

        try:
            # Prepare prompt with healthcare context
            prompt = f"""Healthcare Assistant: I am a healthcare support assistant trained on comprehensive healthcare data.

User: {user_input}

Healthcare Assistant: """

            # Generate response
            outputs = self.llm_pipeline(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Extract generated text
            generated = outputs[0]["generated_text"]
            response = generated.split("Healthcare Assistant: ")[-1].strip()

            # Add safety disclaimer
            if any(
                word in user_input.lower()
                for word in ["medical", "health", "treatment", "diagnosis"]
            ):
                response += "\n\n⚠️ This is general health information only. Always consult qualified healthcare professionals for medical advice."

            return response

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None

    def _create_contextual_response(
        self, user_input: str, category: str
    ) -> Optional[str]:
        """Create contextual response using knowledge base"""
        # Find similar conversation
        similar = self._get_similar_conversation(user_input, category)

        if similar:
            # Adapt the response to current context
            base_response = similar["assistant"]

            # Personalize based on user input
            if "my" in user_input.lower():
                base_response = base_response.replace(
                    "For patients", "For your situation"
                )
                base_response = base_response.replace("Patients", "You")

            # Add variation to avoid exact repetition
            variations = [
                f"Based on your question about {category.replace('_', ' ')}, {base_response}",
                f"{base_response}\n\nFor your specific situation, I'd also recommend discussing this with your healthcare provider.",
                f"Here's what I can share: {base_response}",
                base_response,
            ]

            return random.choice(variations)

        # Fallback to category-based responses
        responses = self.knowledge_base["responses_by_category"].get(category, [])
        if responses:
            return random.choice(responses[:20])  # Choose from top 20 responses

        return None

    def generate_response(self, user_input: str, use_history: bool = True) -> Dict:
        """Generate healthcare response with multiple strategies"""
        start_time = time.time()

        # Detect category
        category = self._detect_category(user_input)

        # Handle crisis immediately
        if category == "crisis":
            return {
                "response": "🚨 CRISIS SUPPORT NEEDED 🚨\n\nImmediate Resources:\n• Call 911 for emergencies\n• National Suicide Prevention Lifeline: 988\n• Crisis Text Line: Text HOME to 741741\n• Local emergency services\n\nYou are not alone. Professional help is available 24/7.\n\n⚠️ If you're in immediate danger, call 911.",
                "category": "crisis",
                "confidence": 1.0,
                "generation_time": time.time() - start_time,
            }

        # Check cache for similar queries
        # Use SHA-256 for security instead of MD5
        input_hash = hashlib.sha256(user_input.lower().encode()).hexdigest()
        if input_hash in self.response_cache:
            cached = self.response_cache[input_hash]
            return {
                **cached,
                "cached": True,
                "generation_time": time.time() - start_time,
            }

        response = None
        confidence = 0.0
        method = "fallback"

        # Try LLM generation first
        if self.use_llm:
            llm_response = self._generate_llm_response(user_input)
            if llm_response and len(llm_response) > 50:
                response = llm_response
                confidence = 0.85
                method = "llm"

        # Fall back to contextual knowledge base
        if not response:
            kb_response = self._create_contextual_response(user_input, category)
            if kb_response:
                response = kb_response
                confidence = 0.75
                method = "knowledge_base"

        # Final fallback
        if not response:
            response = self._get_fallback_response(category)
            confidence = 0.5
            method = "fallback"

        # Add to conversation history
        if use_history:
            self.conversation_history.append(
                {
                    "user": user_input,
                    "assistant": response,
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Cache response
        result = {
            "response": response,
            "category": category,
            "confidence": confidence,
            "method": method,
            "generation_time": time.time() - start_time,
        }

        self.response_cache[input_hash] = result

        return result

    def _get_fallback_response(self, category: str) -> str:
        """Get fallback response for category"""
        fallbacks = {
            "adl": "For activities of daily living support, I recommend consulting with an occupational therapist who can provide personalized assessments and adaptive strategies. They can help with mobility, self-care, and independence. ⚠️ This is general guidance - individual needs vary.",
            "senior_care": "Senior care involves multiple aspects including health monitoring, social engagement, and safety. Consider reaching out to local aging services or geriatric care managers for comprehensive support. ⚠️ Each senior's needs are unique - professional assessment recommended.",
            "mental_health": "Mental health is important. Consider speaking with a mental health professional who can provide appropriate support and strategies. Many resources are available including therapy, support groups, and crisis lines. ⚠️ For mental health concerns, professional guidance is essential.",
            "respite_care": "Caregiver support is crucial. Respite care services, support groups, and temporary relief options are available in most communities. Contact local care agencies for options. ⚠️ Don't hesitate to seek help - caregiver wellbeing is important.",
            "disabilities": "Disability support includes adaptive equipment, accessibility modifications, and community resources. Disability advocacy organizations can provide guidance on rights and services. ⚠️ Consult disability specialists for personalized recommendations.",
            "general": f"I'm a healthcare assistant trained on {self.knowledge_base['total_conversations']:,} conversations. I can help with activities of daily living, senior care, mental health, respite care, and disability support. ⚠️ Always consult healthcare professionals for medical advice.",
        }

        return fallbacks.get(category, fallbacks["general"])

    def get_conversation_stats(self) -> Dict:
        """Get statistics about the AI engine"""
        return {
            "total_training_conversations": self.knowledge_base["total_conversations"],
            "loaded_conversations": sum(
                len(v) for v in self.knowledge_base["conversations"].values()
            ),
            "categories": list(self.knowledge_base["category_keywords"].keys()),
            "llm_enabled": self.use_llm,
            "cache_size": len(self.response_cache),
            "conversation_history": len(self.conversation_history),
        }
