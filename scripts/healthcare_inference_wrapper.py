#!/usr/bin/env python3
"""
Healthcare-aware inference wrapper that provides sensible healthcare responses
"""

import json
import random
import re
from typing import Dict, List


class HealthcareInferenceEngine:
    """Enhanced inference engine for healthcare responses"""

    def __init__(self):
        self.healthcare_knowledge = self.load_healthcare_knowledge()
        self.safety_keywords = [
            "diagnose",
            "treat",
            "cure",
            "medicine",
            "prescription",
            "symptom",
        ]

    def load_healthcare_knowledge(self) -> Dict:
        """Load healthcare knowledge base"""

        return {
            "fitness": {
                "keywords": [
                    "exercise",
                    "fitness",
                    "workout",
                    "training",
                    "gym",
                    "activity",
                    "movement",
                ],
                "responses": [
                    "Start with 10-15 minutes of daily activity like walking or stretching. Gradually increase duration and intensity as you build strength and endurance.",
                    "Focus on consistency over intensity. It's better to exercise 15 minutes daily than 2 hours once a week. Listen to your body and rest when needed.",
                    "Include both cardio and strength training. Aim for 150 minutes of moderate activity weekly, plus 2 strength training sessions.",
                    "Begin with bodyweight exercises like squats, push-ups, and planks. These require no equipment and can be done anywhere.",
                    "Find activities you enjoy - dancing, hiking, swimming, or sports. Exercise shouldn't feel like punishment!",
                    "Set SMART goals: Specific, Measurable, Achievable, Relevant, Time-bound. Track progress with photos or how you feel, not just the scale.",
                ],
                "tips": [
                    "💪 Remember: Every workout counts, no matter how small",
                    "🏃‍♀️ Progress over perfection - celebrate small wins",
                    "⭐ Consistency beats intensity every time",
                    "🎯 Set realistic goals and build sustainable habits",
                ],
            },
            "nutrition": {
                "keywords": [
                    "nutrition",
                    "eating",
                    "food",
                    "meal",
                    "diet",
                    "healthy eating",
                    "cooking",
                ],
                "responses": [
                    "Fill half your plate with vegetables, quarter with lean protein, and quarter with whole grains. This balanced approach supports overall health.",
                    "Stay hydrated by drinking water throughout the day. Add lemon, cucumber, or mint for natural flavor without added sugars.",
                    "Meal prep on weekends can help you make healthier choices during busy weekdays. Prepare nutritious meals and snacks in advance.",
                    "Focus on whole, unprocessed foods. Choose items with fewer ingredients and less added sugar when reading nutrition labels.",
                    "Eat the rainbow! Different colored fruits and vegetables provide different essential nutrients and antioxidants.",
                    "Practice mindful eating: chew slowly, savor flavors, and listen to your body's hunger and fullness cues.",
                ],
                "tips": [
                    "🥗 Fill half your plate with colorful vegetables",
                    "💧 Start your day with a glass of water",
                    "🌈 Eat a variety of colorful fruits and vegetables",
                    "📝 Plan meals and snacks to avoid impulsive choices",
                ],
            },
            "mental_wellness": {
                "keywords": [
                    "stress",
                    "anxiety",
                    "mental",
                    "mindful",
                    "meditation",
                    "relaxation",
                    "mood",
                ],
                "responses": [
                    "Practice deep breathing when feeling overwhelmed: inhale for 4 counts, hold for 4, exhale for 6. This activates your body's relaxation response.",
                    "Start with 5 minutes of meditation daily using apps like Headspace or Calm. Gradually increase duration as it becomes more comfortable.",
                    "Regular exercise, adequate sleep, and mindfulness practices can significantly reduce stress levels and improve mental wellbeing.",
                    "Connect with supportive people, engage in hobbies you enjoy, and set healthy boundaries to protect your mental energy.",
                    "Practice gratitude by writing down 3 things you're grateful for each morning. This can shift your mindset toward positivity.",
                    "It's okay to say no to commitments that drain your energy. Setting boundaries is essential for mental health.",
                ],
                "tips": [
                    "🌬️ Take 5 deep breaths when feeling stressed",
                    "🙏 Practice gratitude daily",
                    "😌 Set healthy boundaries to protect your energy",
                    "🧘‍♀️ Try 5 minutes of meditation daily",
                ],
            },
            "sleep": {
                "keywords": [
                    "sleep",
                    "bedtime",
                    "insomnia",
                    "rest",
                    "tired",
                    "fatigue",
                ],
                "responses": [
                    "Maintain a consistent sleep schedule by going to bed and waking up at the same time daily, even on weekends.",
                    "Create a relaxing bedtime routine: dim lights, avoid screens 1 hour before bed, read, or practice meditation.",
                    "Keep your bedroom cool (65-68°F), dark, and quiet for optimal sleep quality. Consider blackout curtains or a white noise machine.",
                    "Avoid caffeine 6 hours before bedtime and large meals close to sleep time for better rest quality.",
                    "Regular exercise can improve sleep quality, but avoid vigorous workouts within 3 hours of bedtime.",
                    "If you can't fall asleep within 20 minutes, get up and do a quiet activity until you feel sleepy.",
                ],
                "tips": [
                    "😴 Prioritize 7-9 hours of quality sleep nightly",
                    "📚 Create a calming bedtime routine",
                    "🌙 Keep bedroom cool, dark, and quiet",
                    "☕ Avoid caffeine 6 hours before bed",
                ],
            },
            "lifestyle": {
                "keywords": [
                    "routine",
                    "habits",
                    "lifestyle",
                    "wellness",
                    "balance",
                    "self-care",
                ],
                "responses": [
                    "Start your day with a morning routine that energizes you: stretch, hydrate, and set positive intentions for the day.",
                    "Small daily habits create big changes over time. Focus on one new healthy habit at a time for sustainable results.",
                    "Create work-life balance by setting boundaries between professional and personal time. Take actual lunch breaks away from your desk.",
                    "Spend time outdoors daily, even if it's just 10 minutes. Natural light and fresh air can boost mood and energy.",
                    "Declutter your space regularly. An organized environment supports mental clarity and reduces stress.",
                    "Schedule regular check-ins with friends and family. Social connections are vital for emotional wellbeing.",
                ],
                "tips": [
                    "☀️ Create an energizing morning routine",
                    "⚖️ Set boundaries for work-life balance",
                    "🌿 Spend time outdoors daily",
                    "🏡 Keep your space organized and peaceful",
                ],
            },
            "prevention": {
                "keywords": [
                    "prevent",
                    "prevention",
                    "immune",
                    "health",
                    "wellness",
                    "safety",
                ],
                "responses": [
                    "Wash your hands frequently with soap and warm water for at least 20 seconds to prevent illness.",
                    "Support your immune system with colorful fruits, vegetables, adequate sleep, and regular physical activity.",
                    "Wear sunscreen daily with SPF 30+ to protect your skin from UV damage, even on cloudy days.",
                    "Stay up to date with recommended health screenings for your age group. Prevention is better than treatment.",
                    "Practice good posture at your desk and take breaks every 30 minutes to stand and stretch.",
                    "Stay hydrated, especially during exercise and hot weather, to maintain optimal body function.",
                ],
                "tips": [
                    "🧼 Wash hands frequently and properly",
                    "☀️ Wear sunscreen daily",
                    "🩺 Keep up with regular health check-ups",
                    "💧 Stay well-hydrated throughout the day",
                ],
            },
        }

    def identify_category(self, prompt: str) -> str:
        """Identify the healthcare category of the prompt"""

        prompt_lower = prompt.lower()

        for category, data in self.healthcare_knowledge.items():
            if any(keyword in prompt_lower for keyword in data["keywords"]):
                return category

        return "lifestyle"  # Default category

    def check_medical_safety(self, prompt: str) -> bool:
        """Check if prompt is asking for medical advice"""

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.safety_keywords)

    def generate_healthcare_response(
        self, prompt: str, max_length: int = 100, temperature: float = 0.7
    ) -> str:
        """Generate appropriate healthcare response"""

        # Safety check
        if self.check_medical_safety(prompt):
            return (
                "I provide wellness and lifestyle advice, not medical guidance. "
                "For medical concerns, symptoms, or treatment questions, please consult with healthcare professionals."
            )

        # Identify category
        category = self.identify_category(prompt)
        category_data = self.healthcare_knowledge[category]

        # Select response based on temperature (creativity)
        if temperature < 0.4:
            # More focused response
            response = category_data["responses"][0]
        elif temperature > 0.8:
            # More creative, combine tip with response
            response = random.choice(category_data["responses"])
            if random.random() > 0.5 and "tips" in category_data:
                tip = random.choice(category_data["tips"])
                response = f"{response}\n\n{tip}"
        else:
            # Balanced response
            response = random.choice(category_data["responses"])

        # Adjust length
        if max_length < 50:
            # Short response
            words = response.split()
            response = " ".join(words[:15]) + "..."
        elif max_length > 150:
            # Add additional context
            if len(response) < 100 and "tips" in category_data:
                tip = random.choice(category_data["tips"])
                response = f"{response}\n\n{tip}"

        return response

    def get_sample_prompts(self) -> List[str]:
        """Get sample healthcare prompts for testing"""

        return [
            "How to start a fitness routine",
            "Healthy meal prep tips",
            "Natural stress management techniques",
            "Ways to improve sleep quality",
            "Creating a morning wellness routine",
            "Benefits of regular exercise",
            "Healthy eating on a budget",
            "Simple meditation for beginners",
        ]


def test_healthcare_engine():
    """Test the healthcare inference engine"""

    print("🧪 Testing Healthcare Inference Engine")
    print("=" * 40)

    engine = HealthcareInferenceEngine()
    test_prompts = engine.get_sample_prompts()

    for i, prompt in enumerate(test_prompts[:5], 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        response = engine.generate_healthcare_response(
            prompt, max_length=80, temperature=0.7
        )
        print(f"   Response: {response}")

    print("\n✅ Healthcare engine test complete!")


if __name__ == "__main__":
    test_healthcare_engine()
