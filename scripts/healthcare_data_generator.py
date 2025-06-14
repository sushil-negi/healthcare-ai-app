#!/usr/bin/env python3
"""
Non-Medical Healthcare Training Data Generator
Creates comprehensive wellness and lifestyle content for LLM training
"""

import json
import os
import random
from datetime import datetime


class HealthcareDataGenerator:
    """
    Generates training data for non-medical healthcare content
    Focus areas: wellness, fitness, nutrition, mental health, lifestyle
    """

    def __init__(self):
        self.categories = {
            "fitness": "Physical exercise and movement",
            "nutrition": "Healthy eating and dietary advice",
            "mental_wellness": "Mental health and emotional wellbeing",
            "lifestyle": "Healthy lifestyle habits and routines",
            "prevention": "Preventive health measures",
            "wellness_tech": "Health and wellness technology",
        }

    def generate_fitness_content(self):
        """Generate fitness and exercise content"""

        fitness_posts = [
            # Workout motivation
            "Start your day with 10 minutes of movement - even a short walk can boost your energy levels! 💪 #MorningMotivation",
            "Remember: progress over perfection. Every workout counts, no matter how small! 🏃‍♀️ #FitnessJourney",
            "Consistency beats intensity. Better to exercise 15 minutes daily than 2 hours once a week! ⭐",
            "Your body can do it. It's your mind you need to convince. Push through that mental barrier! 🧠💪",
            # Exercise tips
            "Home workout tip: Use water bottles as weights for strength training exercises! 🏠💪",
            "Desk job? Set a timer for every 30 minutes to stand up and stretch. Your back will thank you! ⏰",
            "Try the 7-minute workout app for quick, effective bodyweight exercises anywhere! 📱",
            "Walking is underrated - aim for 10,000 steps daily for improved cardiovascular health! 👟",
            # Specific workouts
            "Quick morning routine: 20 jumping jacks, 15 squats, 10 push-ups, 30-second plank. Ready to conquer the day! 🌅",
            "Strengthen your core with this 5-minute routine: planks, bicycle crunches, mountain climbers, dead bugs! 💪",
            "HIIT workout: 30 seconds high intensity, 30 seconds rest. Repeat 8 rounds. Maximum results, minimum time! ⚡",
            "Yoga flow for beginners: downward dog, child's pose, cat-cow, warrior pose. Hold each for 30 seconds! 🧘‍♀️",
            # Recovery and rest
            "Rest days are not lazy days - they're growth days! Your muscles need time to repair and strengthen! 😴",
            "Post-workout recovery tip: hydrate, stretch, and get quality sleep for optimal muscle recovery! 💧",
            "Listen to your body. Pain is different from discomfort. Know when to push and when to rest! 🎧",
            "Foam rolling after workouts can reduce muscle soreness and improve flexibility! 🔄",
            # Goal setting
            "Set SMART fitness goals: Specific, Measurable, Achievable, Relevant, Time-bound! 🎯",
            "Track your progress with photos, measurements, or how you feel - not just the scale! 📸",
            "Find physical activities you enjoy. Exercise shouldn't feel like punishment! 🎉",
            "Celebrate small wins - completed that first week of workouts? That's amazing progress! 🎊",
        ]

        return fitness_posts

    def generate_nutrition_content(self):
        """Generate nutrition and healthy eating content"""

        nutrition_posts = [
            # Healthy eating basics
            "Fill half your plate with vegetables, quarter with lean protein, quarter with whole grains! 🍽️ #HealthyPlate",
            "Meal prep Sunday: spend 2 hours preparing healthy meals for the entire week! 🥗 #MealPrep",
            "Stay hydrated! Aim for 8 glasses of water daily. Add lemon or cucumber for flavor! 💧",
            "Read nutrition labels: choose foods with fewer ingredients and less added sugar! 🏷️",
            # Superfoods and nutrients
            "Blueberries are brain food! Packed with antioxidants that support cognitive function! 🫐",
            "Avocados provide healthy fats that help your body absorb fat-soluble vitamins! 🥑",
            "Greek yogurt is protein-packed and great for gut health with probiotics! 🥛",
            "Dark leafy greens like spinach and kale are nutrient powerhouses! 🥬",
            # Meal ideas
            "Breakfast idea: overnight oats with berries, nuts, and a drizzle of honey! 🌅🥣",
            "Healthy snack: apple slices with almond butter for fiber, protein, and healthy fats! 🍎",
            "Quick dinner: sheet pan salmon with roasted vegetables - one pan, minimal cleanup! 🐟",
            "Smoothie recipe: spinach, banana, berries, protein powder, and almond milk! 🥤",
            # Mindful eating
            "Practice mindful eating: chew slowly, savor flavors, and listen to hunger cues! 🧘‍♀️",
            "Eat the rainbow! Different colored fruits and vegetables provide different nutrients! 🌈",
            "It's okay to enjoy treats in moderation. Balance and sustainability matter most! ⚖️",
            "Plan your meals and snacks to avoid impulsive food choices when hungry! 📝",
            # Hydration and timing
            "Start your day with a glass of water to kickstart your metabolism! 🌅💧",
            "Eat protein within 30 minutes after workouts to support muscle recovery! 💪",
            "Include fiber-rich foods to support digestive health and keep you feeling full! 🌾",
            "Limit processed foods and choose whole, natural ingredients when possible! 🥕",
        ]

        return nutrition_posts

    def generate_mental_wellness_content(self):
        """Generate mental health and wellness content"""

        mental_wellness_posts = [
            # Stress management
            "Take 5 deep breaths when feeling overwhelmed. Breathe in for 4, hold for 4, out for 6! 🌬️ #StressRelief",
            "Practice gratitude daily: write down 3 things you're grateful for each morning! 🙏 #Gratitude",
            "It's okay to say no. Setting boundaries protects your mental energy and wellbeing! ❌✋",
            "Take breaks from social media. Your mental health will thank you for the digital detox! 📱🚫",
            # Mindfulness and meditation
            "Start with just 5 minutes of meditation daily. Apps like Headspace or Calm can guide you! 🧘‍♂️",
            "Practice mindfulness while walking: notice your surroundings, breath, and footsteps! 👟",
            "Try progressive muscle relaxation: tense and release each muscle group for deep relaxation! 😌",
            "Mindful moments: pause and notice 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste! 👁️",
            # Sleep and rest
            "Prioritize 7-9 hours of quality sleep. Your brain needs this time to recharge and process! 😴",
            "Create a bedtime routine: dim lights, no screens 1 hour before bed, read or meditate! 📚",
            "Keep your bedroom cool, dark, and quiet for optimal sleep quality! 🌙",
            "Avoid caffeine 6 hours before bedtime for better sleep quality! ☕",
            # Emotional wellness
            "It's normal to have bad days. Be kind to yourself and remember tomorrow is a new start! 💙",
            "Connect with others: call a friend, join a group, or volunteer in your community! 🤝",
            "Express yourself through journaling, art, music, or movement! 🎨",
            "Practice self-compassion: treat yourself with the same kindness you'd show a good friend! 💕",
            # Productivity and balance
            "Time blocking: schedule specific times for work, rest, and activities you enjoy! ⏰",
            "Take regular breaks during work: step outside, stretch, or practice breathing exercises! 🌳",
            "Create a morning routine that sets a positive tone for your day! ☀️",
            "End each day by reflecting on one thing that went well! ✨",
        ]

        return mental_wellness_posts

    def generate_lifestyle_content(self):
        """Generate healthy lifestyle content"""

        lifestyle_posts = [
            # Daily habits
            "Small habits, big results: drink water when you wake up, take stairs, park further away! 🚶‍♀️",
            "Create a morning routine that energizes you: stretch, hydrate, set intentions for the day! 🌅",
            "Evening wind-down: dim lights, put away devices, and prepare for restful sleep! 🌙",
            "Sunday planning session: prep meals, organize schedule, and set weekly wellness goals! 📅",
            # Work-life balance
            "Set boundaries between work and personal time. Your wellbeing depends on it! ⚖️",
            "Take actual lunch breaks away from your desk. Your productivity will improve! 🍽️",
            "Use vacation days! Rest and relaxation are essential for long-term performance! 🏖️",
            "Practice saying 'I'll get back to you' instead of immediate yes/no responses! 💭",
            # Social wellness
            "Nurture relationships that energize you and set boundaries with those that drain you! 👥",
            "Schedule regular check-ins with friends and family. Connection is vital for wellbeing! 📞",
            "Join communities aligned with your interests: book clubs, hiking groups, hobby classes! 🏕️",
            "Practice active listening: give full attention when others are speaking! 👂",
            # Environment and space
            "Declutter your space regularly. A organized environment supports mental clarity! 🏡",
            "Bring nature indoors: houseplants improve air quality and mood! 🪴",
            "Create a designated relaxation space in your home, even if it's just a cozy corner! 🛋️",
            "Spend time outdoors daily, even if it's just 10 minutes on your balcony! 🌿",
            # Personal development
            "Read for 20 minutes daily. Choose books that inspire, educate, or entertain you! 📖",
            "Learn something new each month: a skill, language, or hobby that interests you! 🎓",
            "Practice gratitude by keeping a journal of positive experiences and moments! 📔",
            "Set monthly intentions rather than overwhelming yearly resolutions! 🎯",
        ]

        return lifestyle_posts

    def generate_prevention_content(self):
        """Generate preventive health content"""

        prevention_posts = [
            # General prevention
            "Wash your hands frequently and properly - 20 seconds with soap and warm water! 🧼",
            "Get regular check-ups even when feeling healthy. Prevention is better than treatment! 🩺",
            "Wear sunscreen daily, even on cloudy days. Protect your skin from UV damage! ☀️",
            "Stay up to date with recommended health screenings for your age group! 📋",
            # Immune system
            "Support your immune system with colorful fruits, vegetables, and adequate sleep! 🍓",
            "Vitamin D is important: get sunlight exposure or consider supplements after consulting professionals! ☀️",
            "Manage stress levels - chronic stress weakens your immune system! 😤",
            "Stay physically active to boost immune function and overall health! 🏃‍♂️",
            # Safety habits
            "Practice good posture at your desk to prevent back and neck pain! 🪑",
            "Wear proper footwear for activities to prevent injuries! 👟",
            "Stay hydrated during exercise and hot weather to prevent heat-related illness! 💧",
            "Use proper lifting techniques: bend your knees, not your back! 🏋️‍♀️",
            # Mental health prevention
            "Build resilience through regular self-care practices and stress management! 💪",
            "Maintain social connections to support mental health and emotional wellbeing! 🤝",
            "Limit alcohol consumption and avoid smoking for better physical and mental health! 🚭",
            "Practice regular relaxation techniques to prevent burnout and anxiety! 😌",
        ]

        return prevention_posts

    def generate_wellness_tech_content(self):
        """Generate health technology and app content"""

        wellness_tech_posts = [
            # Fitness apps
            "Fitness apps can help track workouts, but remember to listen to your body first! 📱",
            "Use step counters to gamify walking and stay motivated to move more daily! 👟",
            "Try meditation apps for guided practices when you're new to mindfulness! 🧘‍♀️",
            "Food tracking apps can increase awareness of eating patterns and nutrition! 📊",
            # Wearables
            "Wearable devices can motivate activity, but don't become obsessed with the numbers! ⌚",
            "Heart rate monitors help you exercise in appropriate intensity zones! ❤️",
            "Sleep tracking can reveal patterns, but focus on how you feel, not just data! 😴",
            "Use technology as a tool, not a replacement for professional health guidance! 🔧",
            # Digital wellness
            "Practice digital detox: set specific times for checking emails and social media! 📵",
            "Use blue light filters on devices in the evening to protect sleep quality! 💙",
            "Set app usage limits to prevent excessive screen time and promote real-world activities! ⏰",
            "Create tech-free zones in your home, especially the bedroom! 🏠",
            # Health monitoring
            "Regular self-monitoring can help identify patterns in mood, energy, and symptoms! 📈",
            "Use reminder apps for healthy habits like drinking water or taking breaks! 🔔",
            "Virtual wellness classes can provide guidance when in-person options aren't available! 💻",
            "Remember: technology should enhance, not replace, human connection and professional care! 🤝",
        ]

        return wellness_tech_posts

    def generate_conversational_qa(self):
        """Generate Q&A style content for better training"""

        qa_pairs = [
            # Fitness Q&A
            (
                "How do I start exercising if I'm a beginner?",
                "Start with 10-15 minutes of light activity daily. Walking, stretching, or bodyweight exercises are perfect. Gradually increase duration and intensity as you build strength and endurance.",
            ),
            (
                "What's the best time to work out?",
                "The best time is whenever you can be consistent! Some prefer morning workouts for energy, others evening for stress relief. Choose what fits your schedule and preferences.",
            ),
            (
                "How often should I exercise?",
                "Aim for at least 150 minutes of moderate activity weekly, plus 2 strength training sessions. This can be broken into 30-minute sessions, 5 days a week.",
            ),
            # Nutrition Q&A
            (
                "How much water should I drink daily?",
                "Generally 8 glasses (64 oz) daily, but needs vary by activity level, climate, and body size. Your urine should be pale yellow - that's a good hydration indicator.",
            ),
            (
                "What's a healthy snack option?",
                "Combine protein and fiber: apple with almond butter, Greek yogurt with berries, or vegetables with hummus. This keeps you satisfied and provides steady energy.",
            ),
            (
                "How do I meal prep effectively?",
                "Choose one day weekly to plan, shop, and prepare. Cook grains, proteins, and vegetables in batches. Store in containers for easy grab-and-go meals throughout the week.",
            ),
            # Mental wellness Q&A
            (
                "How do I manage stress naturally?",
                "Try deep breathing, regular exercise, adequate sleep, and mindfulness practices. Connect with supportive people and engage in activities you enjoy.",
            ),
            (
                "What's a simple meditation technique?",
                "Start with focused breathing: inhale for 4 counts, hold for 4, exhale for 6. Begin with 5 minutes daily and gradually increase duration as it becomes comfortable.",
            ),
            (
                "How can I improve my sleep quality?",
                "Create a consistent bedtime routine, keep your room cool and dark, avoid screens before bed, and limit caffeine after 2 PM. Regular exercise also promotes better sleep.",
            ),
            # Lifestyle Q&A
            (
                "How do I create healthy habits?",
                "Start small and be consistent. Choose one habit, do it at the same time daily, and track your progress. It takes about 21-66 days to form a new habit.",
            ),
            (
                "What's work-life balance?",
                "Setting boundaries between professional and personal time. This includes taking breaks, using vacation days, and having hobbies outside of work for mental health.",
            ),
            (
                "How do I stay motivated for healthy living?",
                "Set realistic goals, track progress, celebrate small wins, and find activities you enjoy. Having support from friends or communities also helps maintain motivation.",
            ),
        ]

        formatted_qa = []
        for question, answer in qa_pairs:
            formatted_qa.append(f"Q: {question} A: {answer}")
            # Also add as separate statements
            formatted_qa.append(f"Question about wellness: {question}")
            formatted_qa.append(f"Health tip: {answer}")

        return formatted_qa

    def generate_seasonal_content(self):
        """Generate seasonal wellness content"""

        seasonal_posts = [
            # Spring
            "Spring cleaning your habits: what healthy routines will you start this season? 🌸",
            "Seasonal allergies bothering you? Stay hydrated, keep windows closed during high pollen days! 🌼",
            "Take advantage of longer daylight hours for outdoor activities and vitamin D! ☀️",
            # Summer
            "Summer hydration tip: eat water-rich foods like watermelon, cucumber, and leafy greens! 🍉",
            "Protect your skin with SPF 30+ sunscreen, reapplying every 2 hours outdoors! 🧴",
            "Beat the heat: exercise early morning or evening when temperatures are cooler! 🌅",
            # Fall
            "Fall into healthy routines: cozy indoor workouts, warming soups, and earlier bedtimes! 🍂",
            "Seasonal transition: support your immune system with colorful autumn vegetables! 🎃",
            "Shorter days? Create a morning light routine to maintain energy and mood! 💡",
            # Winter
            "Winter wellness: maintain activity levels indoors with yoga, dancing, or bodyweight exercises! ❄️",
            "Combat winter blues: seek light exposure, maintain social connections, practice gratitude! ☀️",
            "Comfort food season: find healthy versions of your favorite warming dishes! 🍲",
        ]

        return seasonal_posts

    def create_comprehensive_dataset(self):
        """Create the complete healthcare training dataset"""

        print("🏥 Generating Non-Medical Healthcare Training Data...")
        print("=" * 60)

        # Generate all content categories
        all_content = []

        print("📋 Generating content categories:")

        # Fitness content
        fitness = self.generate_fitness_content()
        all_content.extend(fitness)
        print(f"  ✅ Fitness: {len(fitness)} posts")

        # Nutrition content
        nutrition = self.generate_nutrition_content()
        all_content.extend(nutrition)
        print(f"  ✅ Nutrition: {len(nutrition)} posts")

        # Mental wellness content
        mental = self.generate_mental_wellness_content()
        all_content.extend(mental)
        print(f"  ✅ Mental Wellness: {len(mental)} posts")

        # Lifestyle content
        lifestyle = self.generate_lifestyle_content()
        all_content.extend(lifestyle)
        print(f"  ✅ Lifestyle: {len(lifestyle)} posts")

        # Prevention content
        prevention = self.generate_prevention_content()
        all_content.extend(prevention)
        print(f"  ✅ Prevention: {len(prevention)} posts")

        # Wellness tech content
        tech = self.generate_wellness_tech_content()
        all_content.extend(tech)
        print(f"  ✅ Wellness Tech: {len(tech)} posts")

        # Q&A content
        qa = self.generate_conversational_qa()
        all_content.extend(qa)
        print(f"  ✅ Q&A Pairs: {len(qa)} items")

        # Seasonal content
        seasonal = self.generate_seasonal_content()
        all_content.extend(seasonal)
        print(f"  ✅ Seasonal: {len(seasonal)} posts")

        # Multiply for more training data
        expanded_content = all_content * 8  # 8x repetition for better learning
        random.shuffle(expanded_content)  # Shuffle for variety

        # Create final dataset
        dataset = {
            "texts": expanded_content,
            "metadata": {
                "domain": "Non-Medical Healthcare",
                "categories": list(self.categories.keys()),
                "total_examples": len(expanded_content),
                "focus_areas": [
                    "Fitness and Exercise",
                    "Nutrition and Healthy Eating",
                    "Mental Health and Wellness",
                    "Healthy Lifestyle Habits",
                    "Preventive Health Measures",
                    "Wellness Technology",
                ],
                "content_types": [
                    "Social media posts",
                    "Health tips and advice",
                    "Q&A pairs",
                    "Motivational content",
                    "Educational information",
                ],
                "target_audience": "General public seeking wellness information",
                "disclaimer": "Content is for educational purposes. Always consult healthcare professionals for medical advice.",
                "generated_date": datetime.now().isoformat(),
            },
        }

        return dataset

    def save_dataset(self, dataset, filename="healthcare_training_data.json"):
        """Save the dataset to file"""

        # Create data directory
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"

        # Save dataset
        with open(filepath, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\n✅ Dataset saved to: {filepath}")
        print(f"📊 Total training examples: {len(dataset['texts']):,}")

        # Show sample content
        print(f"\n📝 Sample training examples:")
        for i, text in enumerate(dataset["texts"][:8], 1):
            print(f"{i}. {text[:80]}...")

        return filepath


def create_training_variations():
    """Create additional training variations"""

    variations = {
        "motivational": [
            "You've got this! Every healthy choice is an investment in your future self! 💪",
            "Progress, not perfection. Small steps lead to big changes over time! 👣",
            "Your health journey is unique. Celebrate your individual wins and progress! 🎉",
            "Consistency beats perfection. Show up for yourself, even on difficult days! ⭐",
        ],
        "educational": [
            "Did you know? Regular physical activity can improve mood and reduce anxiety! 🧠",
            "Health fact: Drinking water first thing in the morning kickstarts your metabolism! 💧",
            "Wellness tip: Deep breathing activates your body's relaxation response! 🌬️",
            "Research shows: Social connections are crucial for mental and physical health! 🤝",
        ],
        "practical": [
            "Quick workout: 10 squats, 10 push-ups, 30-second plank. Repeat 3 times! ⚡",
            "Healthy swap: Replace sugary drinks with water infused with fruit! 🍓",
            "Stress relief: Try the 4-7-8 breathing technique when feeling overwhelmed! 😌",
            "Energy boost: Take a 5-minute walk outside for natural mood enhancement! 🚶‍♀️",
        ],
    }

    return variations


if __name__ == "__main__":
    # Create generator
    generator = HealthcareDataGenerator()

    # Show categories
    print("🏥 NON-MEDICAL HEALTHCARE TRAINING DATA GENERATOR")
    print("=" * 60)
    print("\n📋 Content Categories:")
    for category, description in generator.categories.items():
        print(f"  • {category.title()}: {description}")

    print(f"\n🎯 Content Focus:")
    print("  ✅ Wellness and lifestyle advice")
    print("  ✅ Fitness and exercise tips")
    print("  ✅ Nutrition and healthy eating")
    print("  ✅ Mental health and stress management")
    print("  ✅ Preventive health measures")
    print("  ✅ Health technology and apps")
    print("  ❌ NO medical diagnosis or treatment advice")

    # Generate dataset
    print(f"\n🚀 Generating comprehensive dataset...")
    dataset = generator.create_comprehensive_dataset()

    # Save dataset
    filepath = generator.save_dataset(dataset)

    # Show usage instructions
    print(f"\n🚀 Next Steps:")
    print(f"1. Copy to model: docker cp {filepath} mlops-demo-llm:/app/artifacts/data/")
    print(f"2. Train model: python train_with_healthcare_data.py")
    print(f"3. Test with prompts like:")
    print(f"   - 'How to start a fitness routine'")
    print(f"   - 'Healthy meal prep tips'")
    print(f"   - 'Stress management techniques'")
    print(f"   - 'Morning wellness routine'")

    print(f"\n⚠️  Important Disclaimer:")
    print(f"This content is for educational purposes only.")
    print(f"Always consult healthcare professionals for medical advice.")
