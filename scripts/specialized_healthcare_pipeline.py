#!/usr/bin/env python3
"""
Specialized Healthcare Data Pipeline
Generates 100K+ conversations for each focus area:
- ADL (Activities of Daily Living)
- Senior Care
- Mental Health
- Disabilities Support
- Respite Care
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecializedHealthcarePipeline:
    """Generate specialized healthcare datasets for care services"""

    def __init__(self):
        self.focus_areas = {
            "adl": {
                "name": "Activities of Daily Living",
                "target_conversations": 100000,
                "subcategories": {
                    "mobility": [
                        "walking",
                        "transfers",
                        "wheelchair_use",
                        "fall_prevention",
                        "balance",
                    ],
                    "personal_care": [
                        "bathing",
                        "dressing",
                        "grooming",
                        "toileting",
                        "eating",
                    ],
                    "household_tasks": [
                        "cooking",
                        "cleaning",
                        "laundry",
                        "shopping",
                        "medication_management",
                    ],
                    "communication": [
                        "phone_use",
                        "technology_assistance",
                        "emergency_calls",
                        "social_interaction",
                    ],
                    "safety": [
                        "home_safety",
                        "medical_alerts",
                        "emergency_planning",
                        "fall_detection",
                    ],
                },
            },
            "senior_care": {
                "name": "Senior Care",
                "target_conversations": 100000,
                "subcategories": {
                    "aging_in_place": [
                        "home_modifications",
                        "independent_living",
                        "support_services",
                        "transportation",
                    ],
                    "health_management": [
                        "medication_reminders",
                        "appointment_scheduling",
                        "vital_monitoring",
                        "symptom_tracking",
                    ],
                    "social_engagement": [
                        "loneliness_prevention",
                        "community_activities",
                        "family_connections",
                        "companionship",
                    ],
                    "cognitive_health": [
                        "memory_exercises",
                        "mental_stimulation",
                        "routine_maintenance",
                        "orientation_aids",
                    ],
                    "caregiver_support": [
                        "family_caregiving",
                        "professional_care",
                        "care_coordination",
                        "burnout_prevention",
                    ],
                },
            },
            "mental_health": {
                "name": "Mental Health",
                "target_conversations": 100000,
                "subcategories": {
                    "anxiety_support": [
                        "coping_strategies",
                        "breathing_exercises",
                        "grounding_techniques",
                        "trigger_management",
                    ],
                    "depression_assistance": [
                        "mood_tracking",
                        "activity_scheduling",
                        "social_connection",
                        "professional_referral",
                    ],
                    "stress_management": [
                        "relaxation_techniques",
                        "time_management",
                        "boundary_setting",
                        "self_care",
                    ],
                    "crisis_support": [
                        "emergency_resources",
                        "safety_planning",
                        "immediate_coping",
                        "professional_help",
                    ],
                    "therapy_support": [
                        "session_preparation",
                        "homework_assistance",
                        "progress_tracking",
                        "skill_practice",
                    ],
                },
            },
            "disabilities": {
                "name": "Disabilities Support",
                "target_conversations": 100000,
                "subcategories": {
                    "physical_disabilities": [
                        "mobility_aids",
                        "adaptive_equipment",
                        "accessibility",
                        "physical_therapy",
                    ],
                    "cognitive_disabilities": [
                        "memory_aids",
                        "learning_support",
                        "routine_assistance",
                        "communication_tools",
                    ],
                    "sensory_disabilities": [
                        "visual_aids",
                        "hearing_assistance",
                        "tactile_support",
                        "environmental_modifications",
                    ],
                    "developmental_disabilities": [
                        "life_skills",
                        "social_skills",
                        "employment_support",
                        "independent_living",
                    ],
                    "advocacy": [
                        "rights_awareness",
                        "resource_navigation",
                        "self_advocacy",
                        "legal_support",
                    ],
                },
            },
            "respite_care": {
                "name": "Respite Care",
                "target_conversations": 100000,
                "subcategories": {
                    "caregiver_relief": [
                        "temporary_care",
                        "scheduled_breaks",
                        "emergency_respite",
                        "planned_time_off",
                    ],
                    "care_planning": [
                        "respite_scheduling",
                        "care_instructions",
                        "emergency_contacts",
                        "medical_information",
                    ],
                    "family_support": [
                        "caregiver_stress",
                        "family_dynamics",
                        "communication",
                        "shared_responsibilities",
                    ],
                    "professional_services": [
                        "respite_workers",
                        "care_agencies",
                        "volunteer_programs",
                        "community_resources",
                    ],
                    "quality_assurance": [
                        "care_monitoring",
                        "feedback_systems",
                        "safety_protocols",
                        "service_evaluation",
                    ],
                },
            },
        }

        self.conversation_patterns = [
            "information_seeking",
            "problem_solving",
            "emotional_support",
            "practical_guidance",
            "resource_connection",
            "skill_building",
            "crisis_intervention",
            "care_coordination",
            "advocacy_support",
            "family_guidance",
        ]

        self.user_personas = {
            "care_recipient": {
                "age_groups": ["18-30", "31-50", "51-65", "66-80", "80+"],
                "independence_levels": [
                    "fully_independent",
                    "some_assistance",
                    "significant_support",
                    "full_care",
                ],
                "conditions": [
                    "physical_disability",
                    "cognitive_impairment",
                    "chronic_illness",
                    "aging_related",
                    "mental_health",
                ],
            },
            "family_caregiver": {
                "relationships": [
                    "spouse",
                    "adult_child",
                    "parent",
                    "sibling",
                    "other_family",
                ],
                "experience_levels": [
                    "new_caregiver",
                    "experienced",
                    "long_term",
                    "professional_background",
                ],
                "living_situations": [
                    "same_household",
                    "nearby",
                    "distant",
                    "multiple_caregivers",
                ],
            },
            "professional_caregiver": {
                "roles": [
                    "home_health_aide",
                    "nurse",
                    "therapist",
                    "social_worker",
                    "care_coordinator",
                ],
                "experience_levels": [
                    "entry_level",
                    "experienced",
                    "specialized",
                    "supervisory",
                ],
                "settings": [
                    "home_care",
                    "assisted_living",
                    "community_center",
                    "healthcare_facility",
                ],
            },
        }

    def generate_adl_conversation(
        self, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate ADL-focused conversation"""

        templates = {
            "mobility": {
                "information_seeking": [
                    "I'm having trouble with walking safely. What mobility aids might help me?",
                    "Can you explain different types of walkers and which might be best?",
                    "I need help understanding wheelchair transfer techniques.",
                ],
                "problem_solving": [
                    "I keep losing my balance when getting up from chairs. Any suggestions?",
                    "The stairs in my home are becoming difficult. What options do I have?",
                    "I'm afraid of falling in the bathroom. How can I make it safer?",
                ],
            },
            "personal_care": {
                "practical_guidance": [
                    "I need help with adaptive clothing for someone with limited mobility.",
                    "What are some techniques for safe bathing with mobility limitations?",
                    "Can you suggest tools to help with eating when hands shake?",
                ],
                "emotional_support": [
                    "I feel embarrassed about needing help with personal care.",
                    "How do I maintain dignity while receiving assistance with dressing?",
                    "I'm struggling with accepting help for bathroom needs.",
                ],
            },
        }

        subcat_templates = templates.get(subcategory, templates["mobility"])
        pattern_templates = subcat_templates.get(
            pattern, list(subcat_templates.values())[0]
        )
        user_message = random.choice(pattern_templates)

        # Generate contextual response
        response = self._generate_adl_response(
            user_message, subcategory, pattern, persona
        )

        return self._create_conversation_structure(
            "adl", subcategory, pattern, persona, user_message, response
        )

    def generate_senior_care_conversation(
        self, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Senior Care conversation"""

        templates = {
            "aging_in_place": [
                "What home modifications would help me continue living independently?",
                "I want to stay in my home but need more support. What options exist?",
                "How can I make my home safer as I age?",
                "What transportation options are available for seniors?",
            ],
            "health_management": [
                "I have trouble remembering to take my medications. Any solutions?",
                "How can I better track my health symptoms for doctor visits?",
                "I need help organizing my multiple medical appointments.",
                "What should I monitor daily for my health condition?",
            ],
            "social_engagement": [
                "I've been feeling lonely since my spouse passed. How can I connect with others?",
                "What community activities are good for seniors?",
                "I want to stay connected with my family but don't know how.",
                "How can I find companions my age with similar interests?",
            ],
        }

        messages = templates.get(subcategory, templates["aging_in_place"])
        user_message = random.choice(messages)
        response = self._generate_senior_care_response(
            user_message, subcategory, pattern, persona
        )

        return self._create_conversation_structure(
            "senior_care", subcategory, pattern, persona, user_message, response
        )

    def generate_mental_health_conversation(
        self, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Mental Health conversation"""

        templates = {
            "anxiety_support": [
                "I'm having a panic attack right now. What can I do?",
                "My anxiety is making it hard to leave the house. Any coping strategies?",
                "I need techniques to manage worry about things I can't control.",
                "How can I calm down when I feel overwhelmed?",
            ],
            "depression_assistance": [
                "I'm having trouble getting motivated to do anything. Can you help?",
                "Everything feels overwhelming. Where should I start?",
                "I've lost interest in activities I used to enjoy. Is this normal?",
                "How can I track my mood to better understand my depression?",
            ],
            "crisis_support": [
                "I'm having thoughts of hurting myself. What should I do?",
                "I need immediate mental health support. Where can I get help?",
                "I'm in crisis and don't know who to call.",
                "Someone I know is threatening suicide. How can I help?",
            ],
        }

        messages = templates.get(subcategory, templates["anxiety_support"])
        user_message = random.choice(messages)
        response = self._generate_mental_health_response(
            user_message, subcategory, pattern, persona
        )

        return self._create_conversation_structure(
            "mental_health", subcategory, pattern, persona, user_message, response
        )

    def generate_disabilities_conversation(
        self, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Disabilities Support conversation"""

        templates = {
            "physical_disabilities": [
                "I need help finding adaptive equipment for my home.",
                "What wheelchair accessible modifications can I make?",
                "I'm looking for devices to help with limited hand function.",
                "How can I improve accessibility in my workplace?",
            ],
            "cognitive_disabilities": [
                "I need memory aids to help with daily tasks.",
                "What tools can help with learning and understanding?",
                "I need support creating routines and reminders.",
                "How can I improve my communication abilities?",
            ],
            "advocacy": [
                "I think my disability rights are being violated. What can I do?",
                "How do I navigate disability services and benefits?",
                "I need help advocating for accommodations at work.",
                "What resources exist for disability legal support?",
            ],
        }

        messages = templates.get(subcategory, templates["physical_disabilities"])
        user_message = random.choice(messages)
        response = self._generate_disabilities_response(
            user_message, subcategory, pattern, persona
        )

        return self._create_conversation_structure(
            "disabilities", subcategory, pattern, persona, user_message, response
        )

    def generate_respite_care_conversation(
        self, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Respite Care conversation"""

        templates = {
            "caregiver_relief": [
                "I'm exhausted from caregiving. How can I get temporary help?",
                "I need a break but worry about leaving my loved one. What options exist?",
                "How do I find reliable respite care services?",
                "What should I do in a caregiving emergency when I need immediate help?",
            ],
            "family_support": [
                "My family doesn't understand the stress of caregiving. How can I explain?",
                "I feel guilty about needing breaks from caregiving. Is this normal?",
                "How can we share caregiving responsibilities among family members?",
                "I'm burning out as a caregiver. What support is available?",
            ],
            "professional_services": [
                "How do I evaluate respite care providers?",
                "What questions should I ask potential respite workers?",
                "How much does respite care typically cost?",
                "What's the difference between respite care and home health services?",
            ],
        }

        messages = templates.get(subcategory, templates["caregiver_relief"])
        user_message = random.choice(messages)
        response = self._generate_respite_care_response(
            user_message, subcategory, pattern, persona
        )

        return self._create_conversation_structure(
            "respite_care", subcategory, pattern, persona, user_message, response
        )

    def _generate_adl_response(
        self, user_message: str, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> str:
        """Generate ADL-specific response"""

        disclaimer = "⚠️ This is general ADL guidance. For personalized assessments, consult occupational therapists or healthcare professionals."

        responses = {
            "mobility": "For mobility support, consider these options:\n\n• Mobility aids (walkers, canes, wheelchairs)\n• Home modifications (grab bars, ramps)\n• Physical therapy exercises\n• Fall prevention strategies\n• Balance training\n\n"
            + disclaimer,
            "personal_care": "For personal care assistance:\n\n• Adaptive tools and equipment\n• Modified techniques for independence\n• Caregiver assistance strategies\n• Dignity-preserving approaches\n• Safety considerations\n\n"
            + disclaimer,
            "household_tasks": "For household task management:\n\n• Adaptive kitchen tools\n• Simplified cleaning methods\n• Meal preparation aids\n• Medication organization\n• Energy conservation techniques\n\n"
            + disclaimer,
        }

        return responses.get(subcategory, responses["mobility"])

    def _generate_senior_care_response(
        self, user_message: str, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> str:
        """Generate Senior Care response"""

        disclaimer = "⚠️ This is general senior care information. Consult healthcare providers and aging specialists for personalized guidance."

        responses = {
            "aging_in_place": "For aging in place successfully:\n\n• Home safety assessments\n• Support service coordination\n• Technology solutions\n• Community resources\n• Emergency planning\n\n"
            + disclaimer,
            "health_management": "For effective health management:\n\n• Medication management systems\n• Health monitoring tools\n• Care coordination\n• Preventive care scheduling\n• Emergency preparedness\n\n"
            + disclaimer,
            "social_engagement": "For social connection and engagement:\n\n• Community senior programs\n• Volunteer opportunities\n• Social groups and clubs\n• Technology for connection\n• Intergenerational activities\n\n"
            + disclaimer,
        }

        return responses.get(subcategory, responses["aging_in_place"])

    def _generate_mental_health_response(
        self, user_message: str, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> str:
        """Generate Mental Health response"""

        crisis_disclaimer = "🚨 If you're in immediate danger, call 911. For mental health crisis: National Suicide Prevention Lifeline 988."
        general_disclaimer = "⚠️ This is supportive information only. For mental health concerns, please consult qualified mental health professionals."

        if "crisis" in subcategory or any(
            word in user_message.lower()
            for word in ["suicide", "hurt myself", "crisis", "emergency"]
        ):
            return f"🚨 CRISIS SUPPORT NEEDED 🚨\n\nImmediate Resources:\n• Call 911 for emergencies\n• National Suicide Prevention Lifeline: 988\n• Crisis Text Line: Text HOME to 741741\n• Local emergency services\n\nYou are not alone. Professional help is available 24/7.\n\n{crisis_disclaimer}"

        responses = {
            "anxiety_support": "For anxiety management:\n\n• Deep breathing exercises\n• Grounding techniques (5-4-3-2-1 method)\n• Progressive muscle relaxation\n• Mindfulness practices\n• Professional therapy resources\n\n"
            + general_disclaimer,
            "depression_assistance": "For depression support:\n\n• Small, manageable daily goals\n• Mood tracking tools\n• Social connection strategies\n• Physical activity when possible\n• Professional counseling resources\n\n"
            + general_disclaimer,
            "stress_management": "For stress management:\n\n• Relaxation techniques\n• Time management strategies\n• Boundary setting\n• Self-care practices\n• Support network activation\n\n"
            + general_disclaimer,
        }

        return responses.get(subcategory, responses["anxiety_support"])

    def _generate_disabilities_response(
        self, user_message: str, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> str:
        """Generate Disabilities Support response"""

        disclaimer = "⚠️ This is general disability support information. Consult disability specialists and advocacy organizations for personalized guidance."

        responses = {
            "physical_disabilities": "For physical disability support:\n\n• Adaptive equipment resources\n• Accessibility modifications\n• Assistive technology\n• Physical therapy options\n• Independent living services\n\n"
            + disclaimer,
            "cognitive_disabilities": "For cognitive disability support:\n\n• Memory and learning aids\n• Routine development tools\n• Communication assistance\n• Skills training programs\n• Support services\n\n"
            + disclaimer,
            "advocacy": "For disability advocacy:\n\n• Know your rights under ADA\n• Contact disability advocacy organizations\n• Document accommodation needs\n• Seek legal counsel if needed\n• Connect with peer support groups\n\n"
            + disclaimer,
        }

        return responses.get(subcategory, responses["physical_disabilities"])

    def _generate_respite_care_response(
        self, user_message: str, subcategory: str, pattern: str, persona: Dict[str, Any]
    ) -> str:
        """Generate Respite Care response"""

        disclaimer = "⚠️ This is general respite care information. Consult care coordinators and local agencies for specific services."

        responses = {
            "caregiver_relief": "For caregiver relief:\n\n• Local respite care services\n• Family and volunteer support\n• Professional respite providers\n• Emergency respite options\n• Caregiver support groups\n\n"
            + disclaimer,
            "family_support": "For family caregiver support:\n\n• Caregiver stress management\n• Family communication strategies\n• Shared care planning\n• Support group resources\n• Professional counseling\n\n"
            + disclaimer,
            "professional_services": "For professional respite services:\n\n• Licensed care providers\n• Background-checked workers\n• Specialized care training\n• Insurance coverage options\n• Quality assurance protocols\n\n"
            + disclaimer,
        }

        return responses.get(subcategory, responses["caregiver_relief"])

    def _create_conversation_structure(
        self,
        focus_area: str,
        subcategory: str,
        pattern: str,
        persona: Dict[str, Any],
        user_message: str,
        response: str,
    ) -> Dict[str, Any]:
        """Create standardized conversation structure"""

        return {
            "conversation_id": f"{focus_area}_{subcategory}_{int(time.time())}_{random.randint(1000, 9999)}",
            "focus_area": focus_area,
            "subcategory": subcategory,
            "pattern": pattern,
            "persona": persona,
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "metadata": {
                "safety_level": "care_guidance",
                "professional_disclaimer": True,
                "crisis_protocols": "crisis" in subcategory,
                "source": "specialized_healthcare",
                "version": "3.0.0",
            },
        }

    def generate_focus_area_dataset(
        self, focus_area: str, target_size: int = 100000
    ) -> List[Dict[str, Any]]:
        """Generate large dataset for specific focus area"""

        logger.info(f"Generating {target_size} conversations for {focus_area}...")

        area_config = self.focus_areas[focus_area]
        conversations = []
        generated = 0

        # Get generation function
        generation_functions = {
            "adl": self.generate_adl_conversation,
            "senior_care": self.generate_senior_care_conversation,
            "mental_health": self.generate_mental_health_conversation,
            "disabilities": self.generate_disabilities_conversation,
            "respite_care": self.generate_respite_care_conversation,
        }

        generate_func = generation_functions[focus_area]

        while generated < target_size:
            # Select random subcategory
            subcategories = list(area_config["subcategories"].keys())
            subcategory = random.choice(subcategories)

            # Select conversation pattern
            pattern = random.choice(self.conversation_patterns)

            # Select persona
            persona_type = random.choice(list(self.user_personas.keys()))
            persona = self._generate_persona(persona_type)

            # Generate conversation
            conversation = generate_func(subcategory, pattern, persona)
            conversations.append(conversation)

            generated += 1

            if generated % 5000 == 0:
                logger.info(
                    f"Generated {generated}/{target_size} {focus_area} conversations..."
                )

        logger.info(
            f"Completed {focus_area} dataset: {len(conversations)} conversations"
        )
        return conversations

    def _generate_persona(self, persona_type: str) -> Dict[str, Any]:
        """Generate realistic persona for conversation"""

        persona_config = self.user_personas[persona_type]

        if persona_type == "care_recipient":
            return {
                "type": persona_type,
                "age_group": random.choice(persona_config["age_groups"]),
                "independence_level": random.choice(
                    persona_config["independence_levels"]
                ),
                "condition": random.choice(persona_config["conditions"]),
            }
        elif persona_type == "family_caregiver":
            return {
                "type": persona_type,
                "relationship": random.choice(persona_config["relationships"]),
                "experience_level": random.choice(persona_config["experience_levels"]),
                "living_situation": random.choice(persona_config["living_situations"]),
            }
        else:  # professional_caregiver
            return {
                "type": persona_type,
                "role": random.choice(persona_config["roles"]),
                "experience_level": random.choice(persona_config["experience_levels"]),
                "setting": random.choice(persona_config["settings"]),
            }

    def generate_all_focus_areas(
        self, conversations_per_area: int = 100000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate datasets for all focus areas"""

        logger.info(
            f"Starting generation of {conversations_per_area} conversations per focus area..."
        )
        logger.info(
            f"Total target: {conversations_per_area * len(self.focus_areas)} conversations"
        )

        all_datasets = {}

        for focus_area in self.focus_areas.keys():
            logger.info(f"\n=== Starting {focus_area.upper()} dataset generation ===")
            dataset = self.generate_focus_area_dataset(
                focus_area, conversations_per_area
            )
            all_datasets[focus_area] = dataset

            # Save individual focus area dataset
            self.save_focus_area_dataset(focus_area, dataset)

        return all_datasets

    def save_focus_area_dataset(
        self, focus_area: str, conversations: List[Dict[str, Any]]
    ):
        """Save dataset for individual focus area"""

        # Create output filename
        output_path = f"/Users/snegi/Documents/github/mlops-project/data/{focus_area}_healthcare_data.json"
        jsonl_path = f"/Users/snegi/Documents/github/mlops-project/data/{focus_area}_healthcare_data.jsonl"
        stats_path = f"/Users/snegi/Documents/github/mlops-project/data/{focus_area}_healthcare_stats.json"

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(conversations, f, indent=2)

        # Save as JSONL for training
        with open(jsonl_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        # Generate and save statistics
        stats = self._generate_focus_area_stats(focus_area, conversations)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved {focus_area} dataset:")
        logger.info(f"  JSON: {output_path}")
        logger.info(f"  JSONL: {jsonl_path}")
        logger.info(f"  Stats: {stats_path}")

    def _generate_focus_area_stats(
        self, focus_area: str, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate statistics for focus area dataset"""

        subcategories = {}
        patterns = {}
        persona_types = {}

        for conv in conversations:
            # Subcategory distribution
            subcat = conv.get("subcategory", "unknown")
            subcategories[subcat] = subcategories.get(subcat, 0) + 1

            # Pattern distribution
            pattern = conv.get("pattern", "unknown")
            patterns[pattern] = patterns.get(pattern, 0) + 1

            # Persona distribution
            persona_type = conv.get("persona", {}).get("type", "unknown")
            persona_types[persona_type] = persona_types.get(persona_type, 0) + 1

        return {
            "focus_area": focus_area,
            "total_conversations": len(conversations),
            "generation_timestamp": datetime.now().isoformat(),
            "distribution": {
                "subcategories": subcategories,
                "conversation_patterns": patterns,
                "persona_types": persona_types,
            },
            "safety_features": {
                "crisis_protocol_conversations": sum(
                    1
                    for c in conversations
                    if c.get("metadata", {}).get("crisis_protocols", False)
                ),
                "professional_disclaimer_coverage": sum(
                    1
                    for c in conversations
                    if c.get("metadata", {}).get("professional_disclaimer", False)
                ),
            },
            "version": "3.0.0",
        }


def main():
    """Generate specialized healthcare datasets - one focus area at a time"""

    pipeline = SpecializedHealthcarePipeline()

    # Focus on Disabilities Support - final focus area
    focus_area = "disabilities"
    target_size = 100000  # Full production size for Disabilities Support

    logger.info("🏥 Starting Disabilities Support Healthcare Data Generation")
    logger.info(f"Target: {target_size} conversations for {focus_area}")
    logger.info(f"Focus area: Disabilities Support (FINAL)")

    # Generate Disabilities Support dataset
    logger.info(f"\n=== Starting {focus_area.upper()} dataset generation ===")
    disabilities_dataset = pipeline.generate_focus_area_dataset(focus_area, target_size)

    # Save Disabilities Support dataset
    pipeline.save_focus_area_dataset(focus_area, disabilities_dataset)

    logger.info(f"\n✅ Disabilities Support Generation Complete!")
    logger.info(f"📊 Total conversations generated: {len(disabilities_dataset)}")
    logger.info(f"📁 Files saved to: /Users/snegi/Documents/github/mlops-project/data/")
    logger.info(f"  {focus_area}: {len(disabilities_dataset)} conversations")
    logger.info(
        f"\n🎉 ALL 5 FOCUS AREAS COMPLETE! Total: 500,000 specialized healthcare conversations"
    )


if __name__ == "__main__":
    main()
