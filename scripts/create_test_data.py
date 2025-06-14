#!/usr/bin/env python3
"""
Create test data for MLOps pipeline demonstration
"""

import json
import random


def generate_response(category, query=""):
    """Generate appropriate response based on category"""

    # Crisis response must include emergency resources
    if category == "crisis_mental_health" or any(
        word in query.lower()
        for word in ["suicide", "kill", "hurt myself", "end my life"]
    ):
        return (
            "🚨 CRISIS SUPPORT NEEDED 🚨\n\n"
            "If you're having thoughts of suicide or self-harm, please reach out for help immediately:\n\n"
            "• National Suicide Prevention Lifeline: 988\n"
            "• Crisis Text Line: Text HOME to 741741\n"
            "• Emergency Services: 911\n\n"
            "You matter, and help is available 24/7. Please don't hesitate to reach out to a mental health professional, "
            "trusted friend, or family member. Consider contacting your healthcare provider for additional support."
        )

    # Medical-related responses should include disclaimers
    medical_categories = [
        "senior_medication",
        "mental_health_anxiety",
        "mental_health_depression",
    ]
    if category in medical_categories or any(
        word in query.lower()
        for word in ["medication", "medical", "treatment", "symptoms"]
    ):
        return (
            f"Here's guidance for {category.replace('_', ' ')}:\n\n"
            "• Establish a consistent routine\n"
            "• Use medication reminders or pill organizers\n"
            "• Keep a symptom journal\n"
            "• Stay connected with support systems\n\n"
            "Please consult with your healthcare provider or medical professional for personalized advice. "
            "This information is for educational purposes only and should not replace professional medical guidance."
        )

    # ADL and mobility responses
    if "adl" in category:
        return (
            f"I understand that {category.replace('_', ' ')} can be challenging. "
            "You're taking a positive step by seeking support.\n\n"
            f"Here's helpful guidance for {category.replace('_', ' ')}:\n\n"
            "• Consider adaptive equipment like grab bars or transfer boards to help you\n"
            "• Break tasks into smaller, manageable steps that work for you\n"
            "• Ensure proper lighting and clear pathways for your safety\n"
            "• Practice exercises to maintain your strength and flexibility\n\n"
            "Remember, help is available and you don't have to face this alone. "
            "Please consult with an occupational therapist or healthcare provider for personalized recommendations."
        )

    # Caregiver support responses
    if "caregiver" in category:
        return (
            f"I understand the challenges you're facing with {category.replace('_', ' ')}. "
            "Your dedication is admirable, and you deserve support too.\n\n"
            f"Here's support for {category.replace('_', ' ')}:\n\n"
            "• Take regular breaks to prevent burnout - you need and deserve rest\n"
            "• Join caregiver support groups where you can connect with others who understand\n"
            "• Use respite care services when available - it's okay to ask for help\n"
            "• Practice self-care activities that bring you joy and peace\n\n"
            "Remember: caring for yourself enables you to better care for others. You're not alone in this journey. "
            "Please consider consulting with a healthcare professional for additional support resources."
        )

    # Disability support responses
    if "disability" in category:
        return (
            f"Support for {category.replace('_', ' ')}:\n\n"
            "• Research available assistive technologies\n"
            "• Connect with disability advocacy organizations\n"
            "• Understand your rights under the ADA\n"
            "• Explore community resources and support programs\n\n"
            "Contact disability services or a healthcare provider for personalized assistance and resources."
        )

    # Senior/social responses
    if "senior" in category and "medication" not in category:
        return (
            f"Guidance for {category.replace('_', ' ')}:\n\n"
            "• Join senior community centers or groups\n"
            "• Use technology to stay connected with family\n"
            "• Participate in activities that interest you\n"
            "• Consider volunteer opportunities\n\n"
            "Consult with your healthcare provider about programs and services available in your area."
        )

    # Default response with disclaimer
    return (
        f"I understand you're looking for help with {category.replace('_', ' ')}. "
        "You're not alone in this journey, and support is available.\n\n"
        f"Here's guidance for {category.replace('_', ' ')}:\n\n"
        "• Maintain regular communication with your healthcare providers\n"
        "• Follow your prescribed treatment plans\n"
        "• Keep detailed health records for your reference\n"
        "• Stay informed about your condition\n\n"
        "Remember, you matter and help is available. "
        "Please consult with a qualified healthcare professional for personalized medical advice. "
        "We're here to support you every step of the way."
    )


# Create mock healthcare training data
test_data = []

# Sample queries by category with variations
healthcare_queries = {
    "adl_mobility": [
        "My elderly father has trouble getting out of bed",
        "Grandma struggles with walking to the bathroom",
        "Dad can't stand up from his chair anymore",
        "Mom needs help transferring from wheelchair to bed",
        "My parent has difficulty climbing stairs",
        "Elderly mother can't get up after falling",
        "Father struggles with balance when walking",
        "Grandmother needs assistance with mobility",
        "Parent has trouble moving around the house",
        "Senior family member can't walk without support",
        "Elderly relative needs help with basic movements",
        "Mom can't get in and out of the car",
        "Dad struggles with getting dressed standing up",
        "Parent needs mobility aids for daily activities",
        "Grandparent has limited range of motion",
        "Family member experiences frequent falls",
        "Elderly parent needs physical therapy exercises",
        "Mother requires walker for stability",
        "Father needs grab bars installed",
        "Senior relative needs mobility assessment",
    ],
    "senior_medication": [
        "I need help with medication reminders",
        "Dad keeps forgetting to take his pills",
        "Mom gets confused about her prescriptions",
        "How to organize multiple medications?",
        "Senior parent mixing up medicine doses",
        "Need pill organizer recommendations",
        "Elderly mother resistant to taking meds",
        "Father taking wrong medication times",
        "Managing diabetes medications for elderly",
        "Blood pressure medication schedule help",
        "Pain medication management for seniors",
        "Medication side effects in elderly",
        "Drug interactions worry me",
        "Prescription refill reminders needed",
        "Generic vs brand medications for seniors",
        "Medicine adherence strategies needed",
        "Medication costs too high for parent",
        "Injectable medication assistance required",
        "Pharmacy delivery options for elderly",
        "Medication list organization tips",
    ],
    "caregiver_burnout": [
        "I feel overwhelmed caring for my mother",
        "Exhausted from 24/7 caregiving duties",
        "No time for myself anymore",
        "Feeling resentful about caregiving role",
        "Crying daily from caregiver stress",
        "Lost my identity as a caregiver",
        "Siblings won't help with parent care",
        "Work-life balance impossible with caregiving",
        "Depression from caregiving responsibilities",
        "Angry at parent's demanding behavior",
        "Sleep deprived from night care duties",
        "Financial stress from caregiving costs",
        "Marriage suffering due to caregiving",
        "Friends don't understand caregiver stress",
        "Guilt about caregiver frustrations",
        "Physical health declining from stress",
        "Emotional exhaustion from dementia care",
        "Isolated and lonely as caregiver",
        "Considering quitting job for caregiving",
        "Need emotional support for caregiving",
    ],
    "senior_social": [
        "What exercises are safe for seniors?",
        "Social activities for isolated elderly parent",
        "Senior center programs in my area",
        "Virtual socializing options for elderly",
        "Depression from senior isolation",
        "Making friends in retirement community",
        "Hobbies suitable for limited mobility",
        "Volunteer opportunities for seniors",
        "Dating advice for widowed parent",
        "Technology classes for elderly",
        "Book clubs for senior citizens",
        "Gentle yoga for older adults",
        "Walking groups for seniors",
        "Art therapy for elderly",
        "Music programs for memory care",
        "Gardening with physical limitations",
        "Pet therapy benefits for seniors",
        "Intergenerational activity ideas",
        "Travel options for elderly",
        "Online communities for seniors",
    ],
    "mental_health_anxiety": [
        "I can't stop feeling anxious",
        "Panic attacks getting worse lately",
        "Constant worry about everything",
        "Heart racing from anxiety daily",
        "Can't sleep due to anxious thoughts",
        "Anxiety affecting my work performance",
        "Social anxiety preventing friendships",
        "Health anxiety causing daily distress",
        "Anxiety about aging parents' health",
        "Fear of leaving house increasing",
        "Anxiety medication concerns me",
        "Breathing exercises not helping anxiety",
        "Anxiety worse in morning hours",
        "Catastrophic thinking won't stop",
        "Anxiety about financial future",
        "Nervous about medical appointments",
        "Anxiety attacks during driving",
        "Worried about anxiety affecting family",
        "Need anxiety coping strategies",
        "Anxiety therapy options needed",
    ],
    "mental_health_depression": [
        "I feel sad and hopeless",
        "No motivation to do anything",
        "Everything feels pointless lately",
        "Can't enjoy things anymore",
        "Sleeping all day from depression",
        "Appetite gone due to sadness",
        "Crying spells won't stop",
        "Feel like a burden to everyone",
        "Dark thoughts increasing daily",
        "Isolated myself from loved ones",
        "Work suffering from depression",
        "Nothing brings joy anymore",
        "Feel empty and numb inside",
        "Depression after losing spouse",
        "Seasonal depression getting severe",
        "Medication for depression questions",
        "Depression affecting relationships badly",
        "Want to feel normal again",
        "Depression from chronic illness",
        "Need hope and support",
    ],
    "crisis_mental_health": [
        "I want to hurt myself",
        "Thinking about ending everything",
        "Don't want to live anymore",
        "Planning to kill myself",
        "Life not worth living",
        "Better off dead thoughts",
        "Suicide seems only option",
        "Can't take the pain anymore",
        "Want to disappear forever",
        "Having self-harm urges",
        "Cutting myself for relief",
        "Overdose thoughts won't leave",
        "Bridge seems inviting lately",
        "Gun in house scares me",
        "Pills look tempting today",
        "Wrote goodbye letters already",
        "Family better without me",
        "Nothing left to live for",
        "End suffering permanently",
        "Crisis getting worse hourly",
    ],
    "disability_equipment": [
        "Need help with wheelchair transfers",
        "Mobility scooter recommendations needed",
        "Adaptive equipment for one-handed use",
        "Bathroom modifications for disability",
        "Vehicle modifications for wheelchair",
        "Hearing aid selection guidance",
        "Communication devices for speech issues",
        "Prosthetic limb adjustment problems",
        "Walker vs rollator decision",
        "Stairlift installation questions",
        "Hospital bed for home use",
        "Pressure relief cushion options",
        "Grab bars placement advice",
        "Ramp specifications for home",
        "Adaptive clothing recommendations",
        "Assistive technology for vision",
        "Transfer board usage tips",
        "Hoyer lift operation help",
        "Adaptive eating utensils needed",
        "Emergency alert system options",
    ],
    "disability_rights": [
        "How do I get disability accommodations?",
        "Workplace discrimination due to disability",
        "ADA rights being violated",
        "School refusing accommodation requests",
        "Housing accessibility requirements",
        "Service animal access denied",
        "Disability benefits application help",
        "Employment rights with disability",
        "Reasonable accommodation examples needed",
        "Transportation services for disabled",
        "Voting accessibility concerns",
        "Healthcare discrimination experienced",
        "Insurance denying disability claims",
        "Legal resources for disability rights",
        "Accessibility lawsuit information",
        "Disability advocacy organizations",
        "Public accommodation requirements",
        "Digital accessibility needs",
        "Travel rights with disabilities",
        "Disability disclosure concerns",
    ],
    "caregiver_respite": [
        "I need a break from caregiving",
        "Respite care options available?",
        "Temporary care for vacation",
        "Weekend break desperately needed",
        "Adult daycare information required",
        "In-home respite services cost",
        "Emergency respite care available?",
        "Overnight respite care options",
        "Respite voucher programs exist?",
        "Family refuses respite help",
        "Guilt about taking breaks",
        "Respite care quality concerns",
        "Finding trusted respite providers",
        "Medicaid cover respite care?",
        "Respite for dementia patients",
        "Short-term facility placement",
        "Volunteer respite programs available",
        "Planning respite care schedule",
        "Affordable respite solutions needed",
        "Self-care during respite time",
    ],
    "adl_self_care": [
        "Help with bathing elderly parent",
        "Dressing assistance techniques needed",
        "Toileting help for disabled adult",
        "Feeding assistance for tremors",
        "Grooming help for arthritis",
        "Dental care for resistant senior",
        "Shower safety equipment needed",
        "Incontinence management strategies",
        "Skin care for bedridden patient",
        "Hair washing solutions needed",
        "Nail care for diabetic senior",
        "Adaptive clothing for easy dressing",
        "Bathroom privacy while assisting",
        "Refusing personal care help",
        "Hygiene routine establishment",
        "Safe transfer techniques needed",
        "Dignity during personal care",
        "Male caregiver female patient",
        "Personal care scheduling tips",
        "Resistance to care strategies",
    ],
}

# Generate test data with unique queries
for category, queries in healthcare_queries.items():
    for query in queries:
        test_data.append(
            {
                "query": query,
                "category": category,
                "response": generate_response(category, query),
            }
        )

# Save test data
with open("data/test_healthcare_training.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Created {len(test_data)} test samples")
print("Categories:", set(item["category"] for item in test_data))
