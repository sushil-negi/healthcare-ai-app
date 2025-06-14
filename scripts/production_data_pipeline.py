#!/usr/bin/env python3
"""
Production Healthcare Data Pipeline
Scales current healthcare data generation for production training
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDataPipeline:
    """Production-ready healthcare data generation pipeline"""

    def __init__(self):
        self.wellness_categories = {
            "fitness": {
                "subcategories": [
                    "cardio",
                    "strength_training",
                    "flexibility",
                    "sports",
                    "home_workouts",
                ],
                "complexity_levels": ["beginner", "intermediate", "advanced"],
                "target_examples": 15000,
            },
            "nutrition": {
                "subcategories": [
                    "meal_planning",
                    "dietary_guidelines",
                    "supplements",
                    "hydration",
                    "weight_management",
                ],
                "complexity_levels": ["basic", "intermediate", "advanced"],
                "target_examples": 15000,
            },
            "mental_health": {
                "subcategories": [
                    "stress_management",
                    "anxiety_coping",
                    "mindfulness",
                    "sleep_hygiene",
                    "work_life_balance",
                ],
                "complexity_levels": ["general", "specific", "crisis_support"],
                "target_examples": 20000,
            },
            "preventive_care": {
                "subcategories": [
                    "health_screening",
                    "vaccination",
                    "safety",
                    "hygiene",
                    "risk_factors",
                ],
                "complexity_levels": ["awareness", "actionable", "detailed"],
                "target_examples": 10000,
            },
            "lifestyle": {
                "subcategories": [
                    "habits",
                    "productivity",
                    "relationships",
                    "hobbies",
                    "personal_growth",
                ],
                "complexity_levels": ["tips", "strategies", "comprehensive"],
                "target_examples": 15000,
            },
            "wellness_technology": {
                "subcategories": [
                    "fitness_apps",
                    "wearables",
                    "health_tracking",
                    "telemedicine",
                    "digital_wellness",
                ],
                "complexity_levels": ["basic", "intermediate", "expert"],
                "target_examples": 10000,
            },
        }

        self.conversation_patterns = [
            "question_answer",
            "advice_seeking",
            "goal_setting",
            "progress_tracking",
            "troubleshooting",
            "information_sharing",
            "motivation_support",
            "habit_formation",
        ]

        self.user_personas = [
            {
                "age_group": "18-25",
                "activity_level": "low",
                "goals": ["weight_loss", "energy"],
            },
            {
                "age_group": "26-35",
                "activity_level": "moderate",
                "goals": ["fitness", "stress_management"],
            },
            {
                "age_group": "36-45",
                "activity_level": "varied",
                "goals": ["health_maintenance", "family_wellness"],
            },
            {
                "age_group": "46-55",
                "activity_level": "moderate",
                "goals": ["disease_prevention", "longevity"],
            },
            {
                "age_group": "56+",
                "activity_level": "low_to_moderate",
                "goals": ["mobility", "chronic_condition_management"],
            },
        ]

    def generate_conversation(
        self, category: str, subcategory: str, complexity: str, persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a realistic wellness conversation"""

        # Select conversation pattern
        pattern = random.choice(self.conversation_patterns)

        # Generate context-aware content
        user_message = self._generate_user_message(
            category, subcategory, complexity, persona, pattern
        )
        assistant_response = self._generate_assistant_response(
            user_message, category, subcategory, complexity
        )

        # Create conversation structure
        conversation = {
            "conversation_id": f"{category}_{subcategory}_{int(time.time())}_{random.randint(1000, 9999)}",
            "category": category,
            "subcategory": subcategory,
            "complexity_level": complexity,
            "pattern": pattern,
            "user_persona": persona,
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "metadata": {
                "safety_level": "wellness_only",
                "medical_disclaimer": True,
                "source": "generated",
                "version": "2.0.0",
            },
        }

        # Add follow-up if appropriate
        if pattern in ["goal_setting", "progress_tracking", "habit_formation"]:
            follow_up = self._generate_follow_up(conversation)
            if follow_up:
                conversation["messages"].extend(follow_up)

        return conversation

    def _generate_user_message(
        self,
        category: str,
        subcategory: str,
        complexity: str,
        persona: Dict[str, Any],
        pattern: str,
    ) -> str:
        """Generate realistic user messages based on context"""

        age_group = persona["age_group"]
        activity_level = persona["activity_level"]
        goals = persona["goals"]

        templates = {
            "fitness": {
                "question_answer": [
                    f"I'm {age_group} years old with {activity_level} activity level. What {subcategory} exercises would help me achieve {random.choice(goals)}?",
                    f"Can you recommend {subcategory} routines for someone who is {activity_level}?",
                    f"I'm interested in {subcategory} but I'm a beginner. Where should I start?",
                ],
                "advice_seeking": [
                    f"I want to improve my {subcategory} but I only have 30 minutes a day. Any suggestions?",
                    f"I've been struggling with {subcategory} consistency. What advice do you have?",
                    f"How can I make {subcategory} more enjoyable and sustainable?",
                ],
                "goal_setting": [
                    f"I want to set realistic {subcategory} goals for the next 3 months. Can you help?",
                    f"What would be an achievable {subcategory} goal for someone with {activity_level} activity level?",
                    f"I'm {age_group} and want to start {subcategory}. What goals should I set?",
                ],
            },
            "nutrition": {
                "question_answer": [
                    f"What are some {subcategory} tips for someone in the {age_group} age group?",
                    f"I'm trying to improve my {subcategory}. What should I focus on?",
                    f"Can you explain the basics of {subcategory} for a beginner?",
                ],
                "advice_seeking": [
                    f"I struggle with {subcategory}. Any practical advice?",
                    f"How can I improve my {subcategory} without major lifestyle changes?",
                    f"I'm confused about {subcategory}. Can you simplify it for me?",
                ],
            },
            "mental_health": {
                "question_answer": [
                    f"What are some effective {subcategory} techniques?",
                    f"Can you suggest {subcategory} strategies for daily use?",
                    f"I'm new to {subcategory}. Where should I begin?",
                ],
                "advice_seeking": [
                    f"I've been dealing with stress lately. How can {subcategory} help?",
                    f"What {subcategory} techniques work best for busy people?",
                    f"I'm looking for simple {subcategory} practices I can start today.",
                ],
            },
        }

        category_templates = templates.get(category, templates["fitness"])
        pattern_templates = category_templates.get(
            pattern, category_templates["question_answer"]
        )

        return random.choice(pattern_templates)

    def _generate_assistant_response(
        self, user_message: str, category: str, subcategory: str, complexity: str
    ) -> str:
        """Generate helpful, safe assistant responses"""

        # Safety disclaimer
        disclaimer = "⚠️ This is general wellness information only. For medical concerns, please consult a healthcare professional."

        response_templates = {
            "fitness": {
                "beginner": [
                    f"Great question about {subcategory}! For beginners, I recommend starting slowly and focusing on proper form. Here are some tips:\n\n1. Start with 2-3 sessions per week\n2. Focus on basic movements\n3. Listen to your body\n4. Gradually increase intensity\n\n{disclaimer}",
                    f"Starting your {subcategory} journey is exciting! Here's a beginner-friendly approach:\n\n• Begin with bodyweight exercises\n• Focus on consistency over intensity\n• Learn proper form first\n• Include rest days\n\n{disclaimer}",
                ],
                "intermediate": [
                    f"For {subcategory}, you can now incorporate more variety and challenge. Consider:\n\n1. Progressive overload principles\n2. Compound movements\n3. Structured programming\n4. Recovery protocols\n\n{disclaimer}",
                    f"To advance your {subcategory}, try:\n\n• Varying rep ranges\n• Adding complexity\n• Tracking progress\n• Setting specific goals\n\n{disclaimer}",
                ],
            },
            "nutrition": {
                "basic": [
                    f"For {subcategory}, focus on these fundamentals:\n\n1. Eat a variety of whole foods\n2. Stay hydrated\n3. Practice portion control\n4. Listen to hunger cues\n\n{disclaimer}",
                    f"Good {subcategory} starts with simple principles:\n\n• Include all food groups\n• Limit processed foods\n• Eat regular meals\n• Enjoy food mindfully\n\n{disclaimer}",
                ]
            },
            "mental_health": {
                "general": [
                    f"For {subcategory}, here are some evidence-based strategies:\n\n1. Practice deep breathing\n2. Maintain regular sleep schedule\n3. Stay physically active\n4. Connect with others\n\n{disclaimer}",
                    f"Effective {subcategory} techniques include:\n\n• Mindfulness meditation\n• Progressive muscle relaxation\n• Journaling\n• Regular exercise\n\n{disclaimer}",
                ]
            },
        }

        category_responses = response_templates.get(
            category, response_templates["fitness"]
        )
        # Map complexity levels to available response categories
        complexity_map = {
            "beginner": "beginner",
            "basic": "beginner",
            "general": "beginner",
            "intermediate": "intermediate",
            "specific": "intermediate",
            "actionable": "intermediate",
            "advanced": "intermediate",
            "detailed": "intermediate",
            "comprehensive": "intermediate",
            "expert": "intermediate",
            "tips": "beginner",
            "strategies": "intermediate",
        }

        mapped_complexity = complexity_map.get(complexity, "beginner")
        available_keys = list(category_responses.keys())
        final_complexity = (
            mapped_complexity
            if mapped_complexity in available_keys
            else available_keys[0]
        )

        complexity_responses = category_responses[final_complexity]

        return random.choice(complexity_responses)

    def _generate_follow_up(self, conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate follow-up messages for multi-turn conversations"""

        follow_ups = [
            {
                "role": "user",
                "content": "Thank you! How often should I do this?",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "role": "assistant",
                "content": "For beginners, I recommend starting 2-3 times per week and gradually increasing as you build consistency. Listen to your body and adjust as needed. ⚠️ This is general wellness information only.",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        return follow_ups if random.random() < 0.3 else []  # 30% chance of follow-up

    def generate_production_dataset(
        self, target_size: int = 100000
    ) -> List[Dict[str, Any]]:
        """Generate large-scale production dataset"""

        logger.info(
            f"Generating production dataset with {target_size} conversations..."
        )

        conversations = []
        generated = 0

        for category, config in self.wellness_categories.items():
            category_target = config["target_examples"]
            category_generated = 0

            logger.info(f"Generating {category_target} conversations for {category}...")

            while category_generated < category_target and generated < target_size:
                subcategory = random.choice(config["subcategories"])
                complexity = random.choice(config["complexity_levels"])
                persona = random.choice(self.user_personas)

                conversation = self.generate_conversation(
                    category, subcategory, complexity, persona
                )
                conversations.append(conversation)

                category_generated += 1
                generated += 1

                if generated % 1000 == 0:
                    logger.info(f"Generated {generated} conversations...")

        logger.info(
            f"Production dataset generation complete: {len(conversations)} conversations"
        )
        return conversations

    def save_dataset(self, conversations: List[Dict[str, Any]], output_path: str):
        """Save dataset in multiple formats"""

        # Save as JSONL for training
        jsonl_path = output_path.replace(".json", ".jsonl")
        with open(jsonl_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        # Save as JSON for analysis
        with open(output_path, "w") as f:
            json.dump(conversations, f, indent=2)

        # Generate summary statistics
        stats = self.generate_statistics(conversations)
        stats_path = output_path.replace(".json", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset saved to {output_path} and {jsonl_path}")
        logger.info(f"Statistics saved to {stats_path}")

    def generate_statistics(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate dataset statistics"""

        total_conversations = len(conversations)
        categories = {}
        complexities = {}
        patterns = {}
        persona_ages = {}

        total_messages = 0

        for conv in conversations:
            # Category stats
            cat = conv["category"]
            categories[cat] = categories.get(cat, 0) + 1

            # Complexity stats
            comp = conv["complexity_level"]
            complexities[comp] = complexities.get(comp, 0) + 1

            # Pattern stats
            pattern = conv["pattern"]
            patterns[pattern] = patterns.get(pattern, 0) + 1

            # Persona stats
            age = conv["user_persona"]["age_group"]
            persona_ages[age] = persona_ages.get(age, 0) + 1

            # Message count
            total_messages += len(conv["messages"])

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": (
                total_messages / total_conversations if total_conversations > 0 else 0
            ),
            "distribution": {
                "categories": categories,
                "complexity_levels": complexities,
                "conversation_patterns": patterns,
                "user_personas": persona_ages,
            },
            "generation_timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
        }


def main():
    """Generate production healthcare dataset"""

    pipeline = ProductionDataPipeline()

    # Generate dataset
    target_size = 25000  # Start with 25k for testing, scale to 100k+
    conversations = pipeline.generate_production_dataset(target_size)

    # Save dataset
    output_path = "/Users/snegi/Documents/github/mlops-project/data/production_healthcare_data.json"
    pipeline.save_dataset(conversations, output_path)

    print(f"✅ Production dataset generated successfully!")
    print(f"📊 Total conversations: {len(conversations)}")
    print(f"💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()
