"""
Healthcare AI Engine using Trained ML Model
Uses the model trained on diverse healthcare responses
"""

import hashlib
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthcareTrainedEngine:
    """Healthcare AI using trained ML model with diverse responses"""

    def __init__(self, model_path=None):
        if model_path is None:
            # Check common locations for the model file
            possible_paths = [
                "/app/model.pkl",  # Docker container path
                "./model.pkl",  # Current directory
                "../model.pkl",  # Parent directory
                "model.pkl",  # Same directory
            ]

            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                model_path = "/app/model.pkl"  # Default fallback

        self.model_path = model_path
        self.model_data = None
        self.pipeline = None
        self.category_mapping = None
        self.healthcare_responses = None
        self.conversation_history = []
        self.response_cache = {}

        # Load the trained model
        self._load_model()

    def _load_model(self):
        """Load the trained model and response data"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model_data = pickle.load(f)

                self.pipeline = self.model_data["pipeline"]
                self.category_mapping = self.model_data["category_mapping"]
                self.healthcare_responses = self.model_data["healthcare_responses"]

                logger.info(
                    f"Loaded trained model with {len(self.category_mapping)} categories"
                )
                logger.info(f"Categories: {list(self.category_mapping.values())}")
            else:
                logger.error(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response(self, user_input: str) -> Dict:
        """Generate healthcare response using trained model"""
        start_time = time.time()

        # Check for crisis first
        if self._is_crisis(user_input):
            return self._get_crisis_response(start_time)

        # Check cache
        try:
            # Try with usedforsecurity parameter (Python 3.9+)
            input_hash = hashlib.md5(
                user_input.lower().encode(), usedforsecurity=False
            ).hexdigest()
        except TypeError:
            # Fallback for older Python versions
            input_hash = hashlib.md5(user_input.lower().encode()).hexdigest()
        if input_hash in self.response_cache:
            cached = self.response_cache[input_hash]
            return {
                **cached,
                "cached": True,
                "generation_time": time.time() - start_time,
            }

        try:
            # Check for specific scenarios that should override ML classification
            contextual_response = self._check_specific_scenarios(user_input)
            if contextual_response:
                result = {
                    "response": contextual_response,
                    "category": "contextual_override",
                    "confidence": 0.95,
                    "method": "contextual_analysis",
                    "generation_time": time.time() - start_time,
                }
                # Cache the contextual response
                self.response_cache[input_hash] = result
                return result

            # Predict category using trained model
            predicted_category_idx = self.pipeline.predict([user_input])[0]
            predicted_category = self.category_mapping[predicted_category_idx]

            # Get confidence score (probability)
            probabilities = self.pipeline.predict_proba([user_input])[0]
            confidence = float(probabilities[predicted_category_idx])

            # Get appropriate responses for the category
            category_responses = self.healthcare_responses.get(predicted_category, [])

            if category_responses:
                # Select response based on context
                response = self._select_contextual_response(
                    user_input, category_responses
                )
                method = "ml_model"
            else:
                # Fallback response
                response = self._get_fallback_response(predicted_category)
                method = "fallback"
                confidence *= 0.5

            # Add to conversation history
            self.conversation_history.append(
                {
                    "user": user_input,
                    "assistant": response,
                    "category": predicted_category,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Prepare result
            result = {
                "response": response,
                "category": predicted_category,
                "confidence": confidence,
                "method": method,
                "generation_time": time.time() - start_time,
            }

            # Cache the response
            self.response_cache[input_hash] = result

            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": self._get_error_response(),
                "category": "error",
                "confidence": 0.0,
                "method": "error",
                "generation_time": time.time() - start_time,
            }

    def _is_crisis(self, text: str) -> bool:
        """Check if the message indicates a crisis"""
        crisis_words = [
            "suicide",
            "kill myself",
            "hurt myself",
            "want to die",
            "end it all",
            "harm myself",
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in crisis_words)

    def _get_crisis_response(self, start_time: float) -> Dict:
        """Get crisis response"""
        crisis_responses = self.healthcare_responses.get("crisis_mental_health", [])
        if crisis_responses:
            response = crisis_responses[0]  # Use the comprehensive crisis response
        else:
            response = "🚨 CRISIS SUPPORT 🚨\n\nCall 911 or 988 (Suicide & Crisis Lifeline) immediately. You are not alone."

        return {
            "response": response,
            "category": "crisis_mental_health",
            "confidence": 1.0,
            "method": "crisis_detection",
            "generation_time": time.time() - start_time,
        }

    def _select_contextual_response(self, user_input: str, responses: List[str]) -> str:
        """Select the most appropriate response based on context and generate specific advice"""
        user_lower = user_input.lower()
        user_words = set(user_lower.split())

        # Enhanced keyword mapping for better context understanding
        mobility_keywords = [
            "bed",
            "chair",
            "standing",
            "walking",
            "transfer",
            "wheelchair",
            "mobility",
            "movement",
            "getting up",
            "sitting down",
        ]
        medication_keywords = [
            "medication",
            "pills",
            "medicine",
            "prescription",
            "drug",
            "dose",
            "reminder",
            "memory",
        ]
        caregiver_keywords = [
            "caring",
            "caregiver",
            "spouse",
            "parent",
            "family",
            "overwhelmed",
            "tired",
            "stress",
            "burden",
        ]
        daily_living_keywords = [
            "bathing",
            "dressing",
            "eating",
            "toileting",
            "grooming",
            "cooking",
            "cleaning",
        ]

        # Score responses with enhanced context matching
        best_response = responses[0]
        best_score = 0

        for response in responses:
            response_lower = response.lower()
            response_words = set(response_lower.split())
            score = 0

            # Basic word intersection
            score += len(user_words.intersection(response_words)) * 2

            # Context-specific scoring
            if any(keyword in user_lower for keyword in mobility_keywords):
                if any(
                    keyword in response_lower
                    for keyword in [
                        "transfer",
                        "mobility",
                        "wheelchair",
                        "bed",
                        "chair",
                        "standing",
                    ]
                ):
                    score += 10

            # Exercise-specific scoring
            if any(
                keyword in user_lower
                for keyword in [
                    "exercise",
                    "exercises",
                    "workout",
                    "fitness",
                    "physical activity",
                ]
            ):
                if any(
                    keyword in response_lower
                    for keyword in [
                        "exercise",
                        "balance",
                        "strength",
                        "tai chi",
                        "walking",
                        "movement",
                    ]
                ):
                    score += 15

            if any(keyword in user_lower for keyword in medication_keywords):
                if any(
                    keyword in response_lower
                    for keyword in [
                        "medication",
                        "pill",
                        "reminder",
                        "organizer",
                        "alarm",
                    ]
                ):
                    score += 10

            if any(keyword in user_lower for keyword in caregiver_keywords):
                if any(
                    keyword in response_lower
                    for keyword in ["respite", "support", "care", "break", "help"]
                ):
                    score += 10

            if any(keyword in user_lower for keyword in daily_living_keywords):
                if any(
                    keyword in response_lower
                    for keyword in ["adaptive", "equipment", "aids", "assistance"]
                ):
                    score += 10
                # Specific daily living activity matching
                if "eating" in user_lower and "eating" in response_lower:
                    score += 20
                if "dressing" in user_lower and "dressing" in response_lower:
                    score += 20
                if "bathing" in user_lower and "bathing" in response_lower:
                    score += 20

            # Specific scenario bonuses
            if "father" in user_lower or "elderly" in user_lower:
                if "senior" in response_lower or "older" in response_lower:
                    score += 5

            if "dementia" in user_lower or "alzheimer" in user_lower:
                if "memory" in response_lower or "cognitive" in response_lower:
                    score += 8

            if score > best_score:
                best_score = score
                best_response = response

        # If still no good match, create a contextual response
        if best_score < 5:
            best_response = self._generate_contextual_response(user_input, responses[0])

        return best_response

    def _check_specific_scenarios(self, user_input: str) -> Optional[str]:
        """Check for specific scenarios that need contextual responses"""
        user_lower = user_input.lower()

        # Mobility and bed-related issues
        if any(
            term in user_lower for term in ["father", "mother", "parent", "elderly"]
        ) and any(
            term in user_lower
            for term in [
                "bed",
                "getting up",
                "standing",
                "trouble",
                "difficulty",
                "struggle",
            ]
        ):
            return "For helping an elderly parent with bed mobility: 1) Install bed rails for grip support, 2) Raise bed height to ease standing, 3) Use a bed assist handle, 4) Place a firm pillow behind back for leverage, 5) Consider a transfer pole for stability. Physical therapy can teach safe transfer techniques. ⚠️ Consult healthcare providers for personalized mobility assessments."

        # Medication organization
        if any(
            term in user_lower
            for term in ["medication", "medications", "pills", "medicine"]
        ) and any(
            term in user_lower
            for term in ["organize", "organizing", "sort", "manage", "arrangement"]
        ):
            return "For organizing medications effectively: 1) Use a weekly pill organizer with AM/PM compartments, 2) Keep medications in original bottles for reference, 3) Create a medication list with dosages and times, 4) Store medications in a cool, dry place away from children, 5) Use color-coded labels for different times of day, 6) Review medications monthly and dispose of expired ones safely. ⚠️ Ask your pharmacist about medication therapy management services."

        # Medication and memory combination
        if "medication" in user_lower and any(
            term in user_lower for term in ["memory", "forget", "remember", "reminder"]
        ):
            return "For medication management with memory issues: 1) Use automated pill dispensers with alarms, 2) Set multiple phone reminders, 3) Ask pharmacy for blister packaging, 4) Create a daily medication checklist, 5) Involve family in medication monitoring, 6) Consider medication synchronization services. ⚠️ Consult pharmacist about comprehensive medication management programs."

        # Caregiver overwhelm scenarios
        if any(
            term in user_lower for term in ["overwhelmed", "exhausted", "burned out"]
        ) and any(
            term in user_lower
            for term in ["caring", "caregiver", "spouse", "husband", "wife", "partner"]
        ):
            return "Caregiver overwhelm is common and valid. Immediate steps: 1) Contact local Area Agency on Aging for respite services, 2) Join caregiver support groups (online or local), 3) Ask family/friends for specific help, 4) Schedule regular breaks for yourself, 5) Consider adult day programs, 6) Look into professional home care services. ⚠️ Your wellbeing directly impacts care quality - seeking help is essential."

        # Wheelchair transfers
        if "wheelchair" in user_lower and any(
            term in user_lower
            for term in ["transfer", "moving", "getting in", "getting out"]
        ):
            return "For safe wheelchair transfers: 1) Lock wheelchair brakes securely, 2) Position at 45° angle to transfer surface, 3) Use transfer board if needed, 4) Keep back straight, pivot with legs, 5) Practice with occupational therapist. Equipment to consider: bed height adjustment, grab bars, transfer belts, sliding boards. ⚠️ Professional training prevents injuries to both patient and caregiver."

        # Mental health specifics
        if any(
            term in user_lower
            for term in ["depressed", "depression", "sad", "hopeless", "down"]
        ):
            return "Depression affects many people and is treatable. Steps to take: 1) Contact your doctor or mental health professional, 2) Maintain daily routines, 3) Stay connected with supportive people, 4) Engage in activities you usually enjoy, 5) Consider counseling or therapy, 6) Monitor sleep and exercise. ⚠️ If having thoughts of self-harm, call 988 (Suicide & Crisis Lifeline) immediately."

        if any(
            term in user_lower
            for term in ["anxious", "anxiety", "worried", "panic", "nervous"]
        ):
            return "For managing anxiety: 1) Practice deep breathing (4 counts in, 6 counts out), 2) Use grounding techniques (5 things you see, 4 you hear, 3 you touch), 3) Maintain regular sleep and exercise, 4) Limit caffeine and alcohol, 5) Consider mindfulness apps or therapy, 6) Try progressive muscle relaxation. ⚠️ For persistent anxiety, professional help can provide effective treatment options."

        # Senior exercise requests
        if ("exercise" in user_lower or "exercises" in user_lower) and (
            "senior" in user_lower or "elderly" in user_lower or "older" in user_lower
        ):
            return "Safe exercises for seniors: 1) Chair exercises for strength (seated leg lifts, arm raises), 2) Walking - start with 10 minutes daily, 3) Balance work - heel-to-toe walking near wall, 4) Gentle stretching for flexibility, 5) Water aerobics for low-impact cardio, 6) Tai chi for balance and coordination. Always consult healthcare provider before starting new exercise. ⚠️ Start slowly and progress gradually."

        # Eating equipment/adaptations
        if "eating" in user_lower and any(
            term in user_lower
            for term in ["equipment", "adaptive", "help", "aids", "tools", "need"]
        ):
            return "Eating adaptations for independence: 1) Weighted utensils for hand tremors, 2) Built-up handles for weak grip strength, 3) Plate guards to prevent food spilling, 4) Non-slip mats under plates and bowls, 5) Adaptive cups with lids or straws, 6) Rocker knives for one-handed cutting. Occupational therapists can assess specific needs. ⚠️ Proper adaptive equipment significantly improves dining independence and dignity."

        # Pill reminder systems
        if ("pill" in user_lower and "reminder" in user_lower) or (
            "medication" in user_lower and "memory loss" in user_lower
        ):
            return "Pill reminder systems for memory loss: 1) Automated pill dispensers with alarms and locked compartments, 2) Smartphone apps with multiple daily alerts, 3) Visual medication calendars with pictures, 4) Family member text reminders, 5) Pharmacy auto-refill and reminder services, 6) Smart home devices with voice reminders. ⚠️ For severe memory issues, supervised medication administration may be necessary."

        # Workplace accommodation rights
        if "workplace accommodation" in user_lower and "rights" in user_lower:
            return "Workplace accommodation rights: 1) Employers must provide reasonable accommodations under ADA, 2) Submit written accommodation request to HR, 3) Provide medical documentation if requested, 4) Examples: flexible scheduling, adaptive equipment, remote work, 5) Cannot be discriminated against for requesting accommodations, 6) File EEOC complaint if rights violated. ⚠️ Contact disability rights organizations for free legal advice and advocacy support."

        return None

    def _generate_contextual_response(self, user_input: str, base_response: str) -> str:
        """Generate a more contextual response based on user input"""
        user_lower = user_input.lower()

        # Analyze user input for specific scenarios
        if "father" in user_lower and (
            "bed" in user_lower or "getting up" in user_lower
        ):
            return "For helping an elderly father with bed mobility: 1) Install bed rails for grip support, 2) Raise bed height to ease standing, 3) Use a bed assist handle, 4) Place a firm pillow behind back for leverage. Physical therapy can teach safe transfer techniques. ⚠️ Consult healthcare providers for personalized mobility assessments."

        elif "medication" in user_lower and "memory" in user_lower:
            return "For medication management with memory issues: 1) Use automated pill dispensers with alarms, 2) Set multiple phone reminders, 3) Ask pharmacy for blister packaging, 4) Create a daily medication checklist, 5) Involve family in medication monitoring. ⚠️ Consult pharmacist about medication synchronization services."

        elif "overwhelmed" in user_lower and (
            "spouse" in user_lower or "caring" in user_lower
        ):
            return "Caregiver overwhelm is common and valid. Immediate steps: 1) Contact local Area Agency on Aging for respite services, 2) Join caregiver support groups (online or local), 3) Ask family/friends for specific help, 4) Schedule regular breaks for yourself, 5) Consider adult day programs. ⚠️ Your wellbeing directly impacts care quality - seeking help is essential."

        elif "wheelchair" in user_lower and "transfer" in user_lower:
            return "For safe wheelchair transfers: 1) Lock wheelchair brakes, 2) Position at 45° angle to transfer surface, 3) Use transfer board if needed, 4) Keep back straight, pivot with legs, 5) Practice with occupational therapist. Consider: bed height adjustment, grab bars, transfer belts. ⚠️ Professional training prevents injuries."

        elif (
            "depression" in user_lower
            or "sad" in user_lower
            or "hopeless" in user_lower
        ):
            return "Depression affects many people and is treatable. Steps to take: 1) Contact your doctor or mental health professional, 2) Maintain daily routines, 3) Stay connected with supportive people, 4) Engage in activities you usually enjoy, 5) Consider counseling or therapy. ⚠️ If having thoughts of self-harm, call 988 (Suicide & Crisis Lifeline) immediately."

        elif (
            "anxiety" in user_lower or "worried" in user_lower or "panic" in user_lower
        ):
            return "For managing anxiety: 1) Practice deep breathing (4 counts in, 6 counts out), 2) Use grounding techniques (5 things you see, 4 you hear, 3 you touch), 3) Maintain regular sleep and exercise, 4) Limit caffeine and alcohol, 5) Consider mindfulness apps or therapy. ⚠️ For persistent anxiety, professional help can provide effective treatment options."

        else:
            # Enhance the base response with context
            return f"Based on your specific situation: {base_response}"

    def _get_fallback_response(self, category: str) -> str:
        """Get fallback response for category"""
        fallbacks = {
            "adl_mobility": "For mobility support, I recommend consulting with a physical therapist or occupational therapist who can assess your specific needs and provide personalized strategies. ⚠️ Professional assessment is important for safety.",
            "adl_self_care": "Daily living activities can be adapted in many ways. An occupational therapist can evaluate your specific needs and recommend appropriate aids and techniques. ⚠️ Individual assessment ensures the best solutions.",
            "senior_medication": "Medication management is crucial. Consider using pill organizers, setting reminders, and regular medication reviews with your pharmacist. ⚠️ Always consult healthcare providers about medication concerns.",
            "senior_social": "Social connection is vital for seniors. Local senior centers, community programs, and technology can help maintain relationships. ⚠️ Reach out to local aging services for resources.",
            "mental_health_anxiety": "Anxiety can be managed with various techniques. Consider breathing exercises, mindfulness, and professional support. ⚠️ For persistent anxiety, mental health professionals can provide effective treatment.",
            "mental_health_depression": "Depression is treatable. Maintaining routines, social connections, and seeking professional help are important steps. ⚠️ Mental health professionals can provide appropriate support and treatment.",
            "caregiver_respite": "Caregiver support is essential. Respite care services, support groups, and self-care are important. ⚠️ Contact local caregiving agencies for respite options.",
            "caregiver_burnout": "Caregiver burnout is common and serious. Setting boundaries, asking for help, and self-care are crucial. ⚠️ Don't hesitate to seek support for yourself.",
            "disability_equipment": "Adaptive equipment can greatly improve independence. Consult with assistive technology specialists for assessments. ⚠️ Proper evaluation ensures the right equipment for your needs.",
            "disability_rights": "You have rights to accommodations and accessibility. Contact disability advocacy organizations for guidance. ⚠️ Know your rights under ADA and other disability laws.",
            "crisis_mental_health": "🚨 If you're in crisis, please call 988 or 911 immediately. Help is available 24/7. ⚠️ Your safety is the top priority.",
        }

        return fallbacks.get(
            category,
            "I can help with healthcare questions. Please provide more details about your specific needs. ⚠️ For medical advice, consult healthcare professionals.",
        )

    def _get_error_response(self) -> str:
        """Get error response"""
        return "I apologize, but I'm having difficulty processing your request. Please try rephrasing your question or contact support if the issue persists. ⚠️ For urgent healthcare needs, contact your healthcare provider directly."

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "model_loaded": self.model_data is not None,
            "categories": len(self.category_mapping) if self.category_mapping else 0,
            "category_list": (
                list(self.category_mapping.values()) if self.category_mapping else []
            ),
            "total_responses": (
                sum(len(responses) for responses in self.healthcare_responses.values())
                if self.healthcare_responses
                else 0
            ),
            "cache_size": len(self.response_cache),
            "conversation_history": len(self.conversation_history),
            "model_type": "TfidfVectorizer + MultinomialNB",
        }
