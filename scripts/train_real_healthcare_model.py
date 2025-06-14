#!/usr/bin/env python3
"""
Train a Real Healthcare Model with Diverse Responses
Logs to MLflow with actual training metrics
"""

import json
import os
import pickle
import random
import time
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Healthcare response templates with variations
HEALTHCARE_RESPONSES = {
    "adl_mobility": [
        "For wheelchair transfers, the key techniques include: 1) Lock the wheelchair brakes, 2) Position at 45-degree angle to target surface, 3) Use sliding board if needed, 4) Keep back straight and pivot with legs. Would you like specific guidance for bed, car, or toilet transfers?",
        "Safe walking requires: proper footwear with non-slip soles, clearing pathways of obstacles, adequate lighting especially at night, using handrails on stairs, and considering a mobility aid evaluation. An occupational therapist can assess your specific needs.",
        "Balance exercises that help: heel-to-toe walking, standing on one foot (near support), gentle tai chi movements, seated leg lifts, and ankle pumps. Start with 5-10 minutes daily, always near sturdy support.",
        "Fall prevention strategies: install grab bars in bathroom, use night lights, remove throw rugs, keep frequently used items within reach, wear shoes indoors, and consider a personal emergency response system.",
        "Adaptive equipment for mobility: front-wheeled walkers for Parkinson's, rollators with seats for fatigue, transfer poles beside beds, raised toilet seats, and shower chairs. Medicare often covers prescribed equipment.",
    ],
    "adl_self_care": [
        "Dressing aids that promote independence: button hooks, zipper pulls, sock aids, long-handled shoe horns, elastic shoelaces, and adaptive clothing with velcro or magnetic closures. Front-opening garments are often easier.",
        "Bathing safely: use grab bars, non-slip mats, shower chair or transfer bench, hand-held shower head, long-handled sponge, and soap dispensers. Keep water temperature below 120°F to prevent burns.",
        "Eating adaptations: weighted utensils for tremors, built-up handles for weak grip, plate guards to prevent spilling, non-slip mats under plates, and adaptive cups with lids or straws.",
        "Grooming tools: electric toothbrushes, long-handled combs, suction nail brushes, magnifying mirrors, and lever-handle faucets. These reduce strain and promote independence.",
        "Toileting aids: raised toilet seats (3-6 inches), toilet safety frames, bidets or toilet aids for cleaning, and comfortable grab bar placement at 33-36 inches from floor.",
    ],
    "senior_medication": [
        "Medication management systems: weekly pill organizers with AM/PM compartments, automatic pill dispensers with alarms, medication reminder apps, and pharmacy blister packs. Consider large-print labels.",
        "Preventing medication errors: keep updated medication list, use single pharmacy, review medications regularly with doctor, never stop medications without consulting physician, and understand each medication's purpose.",
        "Managing multiple medications: create a medication schedule chart, set phone alarms, link doses to daily activities (meals, bedtime), use medication tracking apps, and involve family members in reminders.",
        "Safe medication storage: keep in original containers, store in cool dry place (not bathroom), lock up if cognitive concerns, dispose of expired medications at pharmacy take-back programs, never share medications.",
        "Questions for your pharmacist: potential side effects, food/drug interactions, what to do if you miss a dose, generic alternatives, and whether medications can be synchronized for same refill date.",
    ],
    "senior_social": [
        "Combating senior isolation: join senior centers for activities and meals, participate in faith communities, volunteer at libraries or schools, join hobby clubs, or attend community education classes.",
        "Technology for connection: video calls with family (Zoom, FaceTime), online senior communities, virtual museum tours, online games with friends, and tablets designed for seniors with simplified interfaces.",
        "Transportation options: senior shuttle services, volunteer driver programs, rideshare services with senior assistance, public transit training programs, and medical transport services covered by insurance.",
        "Intergenerational programs: reading to children at libraries, mentoring programs, sharing skills/crafts at schools, participating in oral history projects, and joining adopt-a-grandparent programs.",
        "Home-based social activities: hosting tea or coffee hours, starting a book club, organizing card games, sharing meals with neighbors, or participating in telephone reassurance programs.",
    ],
    "mental_health_anxiety": [
        "Immediate anxiety relief techniques: 4-7-8 breathing (inhale 4, hold 7, exhale 8), progressive muscle relaxation starting from toes, grounding with 5 senses, cold water on wrists, and gentle movement.",
        "Daily anxiety management: consistent sleep schedule (7-9 hours), limit caffeine and alcohol, regular exercise (even 10-minute walks help), journaling worries, and practicing gratitude daily.",
        "Cognitive strategies: challenge catastrophic thinking, write down evidence for/against worries, set 'worry time' limits, practice self-compassion, and use positive affirmations that feel authentic.",
        "Professional support options: cognitive behavioral therapy (CBT), acceptance therapy (ACT), medication management with psychiatrist, support groups (in-person or online), and employee assistance programs.",
        "Creating calm environments: declutter living spaces, use calming scents (lavender), play soft background music, adjust lighting to warm tones, and create a designated relaxation corner.",
    ],
    "mental_health_depression": [
        "Depression coping strategies: maintain daily routine even when difficult, set small achievable goals, practice self-care basics (shower, dress), engage in pleasant activities even briefly, and track mood patterns.",
        "Behavioral activation: schedule one enjoyable activity daily, break tasks into tiny steps, celebrate small accomplishments, involve accountability partner, and use activity scheduling apps.",
        "Social support for depression: tell trusted friends how they can help, join depression support groups, consider peer support specialists, use crisis text lines when needed, and maintain some social contact daily.",
        "Physical health and depression: aim for 20 minutes sunlight/daylight, eat regular meals even without appetite, gentle exercise like stretching, maintain sleep hygiene, and consider vitamin D screening.",
        "When to seek immediate help: thoughts of self-harm, sudden worsening of symptoms, inability to care for self, psychotic symptoms, or feeling unsafe. Call 988 for crisis support.",
    ],
    "caregiver_respite": [
        "Respite care options: adult day programs (social and health models), in-home respite workers, overnight respite facilities, hospice respite care, and informal respite through family/friends rotation.",
        "Planning respite breaks: start with short breaks (2-4 hours), gradually increase duration, prepare care recipient for changes, leave detailed care instructions, and use respite time for self-care not just errands.",
        "Finding respite providers: Area Agency on Aging resources, faith community volunteers, care.com or sittercity.com, local college nursing students, and respite voucher programs through Medicaid.",
        "Emergency respite planning: maintain list of backup caregivers, keep care recipient information packet ready, identify 24-hour respite facilities, and inform primary doctor of respite plans.",
        "Making respite affordable: Medicaid waivers, veteran benefits, long-term care insurance, sliding scale programs, respite scholarships, and bartering with other caregiving families.",
    ],
    "caregiver_burnout": [
        "Recognizing burnout signs: constant exhaustion, increased irritability, neglecting own health, feeling resentful, social withdrawal, changes in sleep/appetite, and frequent illness from lowered immunity.",
        "Preventing caregiver burnout: set realistic boundaries, ask for help before crisis, maintain one personal activity weekly, join caregiver support groups, and take micro-breaks throughout day.",
        "Self-care for caregivers: 10-minute meditation apps, chair yoga videos, meal prep on weekends, schedule own medical appointments, maintain one friendship, and practice saying 'no' to additional responsibilities.",
        "Building support network: create care team with family/friends, use scheduling apps for help coordination, accept offered help with specific tasks list, join online caregiver communities, and consider counseling.",
        "Caregiver resources: National Alliance for Caregiving, Family Caregiver Alliance, AARP caregiving resources, local caregiver resource centers, and employer caregiver benefits.",
    ],
    "disability_equipment": [
        "Mobility equipment options: manual wheelchairs (lightweight, standard, sport), power wheelchairs with various controls, scooters for longer distances, standing frames, and gait trainers. Insurance often requires trials.",
        "Communication devices: speech generating devices (SGDs), communication boards, switch-activated devices, eye-gaze systems, and communication apps for tablets. Speech therapists provide assessments.",
        "Home modifications: ramps (1:12 slope ratio), doorway widening (32-36 inches), roll-in showers, adjustable height counters, accessible door handles, and smart home technology for independence.",
        "Adaptive computer access: voice recognition software, alternative keyboards, mouth sticks, eye tracking, switch access, and screen readers. Many are built into operating systems.",
        "Daily living aids: reachers/grabbers, dressing sticks, adapted kitchen tools, environmental control units, transfer boards, and positioning aids. Occupational therapists recommend specific items.",
    ],
    "disability_rights": [
        "ADA accommodations: reasonable workplace modifications, auxiliary aids for communication, accessible parking/entrances, service animal rights, and program accessibility. Request in writing for documentation.",
        "Education rights: IEP/504 plans, assistive technology in schools, accessible testing, related services (therapy), and least restrictive environment. Parents can request evaluations anytime.",
        "Housing rights: reasonable modifications (even in rentals), assistance animals beyond pets, accessible unit requests, and Fair Housing Act protections. Landlords cannot charge extra fees.",
        "Employment protections: disclosure is voluntary, reasonable accommodations process, Job Accommodation Network (JAN) resources, protection from discrimination, and equal opportunity for advancement.",
        "Accessing benefits: SSDI/SSI applications, Medicaid waivers, vocational rehabilitation services, ABLE accounts for savings, and Medicare coverage. Consider benefits counseling before working.",
    ],
    "crisis_mental_health": [
        "🚨 IMMEDIATE CRISIS SUPPORT 🚨\n\nIf you're having thoughts of suicide or self-harm:\n• Call 988 (Suicide & Crisis Lifeline) - 24/7\n• Text 'HELLO' to 741741 (Crisis Text Line)\n• Call 911 or go to nearest emergency room\n• Call your therapist's emergency number\n\nYou are not alone. Trained counselors are available right now to help you through this moment. Your life has value and meaning.",
        "Safety planning: Remove means of harm, identify warning signs, list coping strategies that work, call supportive people, go to safe environments, and keep crisis numbers visible. Share plan with trusted person.",
        "Coping with crisis urges: Use ice cubes instead of self-harm, intense exercise, loud music, drawing on skin with marker, tearing paper, or calling a friend. These provide safer sensory alternatives.",
        "After a crisis: Be gentle with yourself, maintain basic self-care, follow up with mental health provider, adjust medications if needed, and consider intensive outpatient programs for additional support.",
        "Supporting someone in crisis: Stay calm, listen without judgment, don't leave them alone, help them contact professionals, remove harmful items, and follow up. Your presence alone helps.",
    ],
}


def generate_training_data():
    """Generate diverse healthcare training data"""
    training_data = []

    # Create varied questions for each category
    question_templates = {
        "adl_mobility": [
            "How do I transfer from {} to {}?",
            "What equipment helps with {}?",
            "I'm having trouble with {}, what can help?",
            "My {} makes {} difficult, any suggestions?",
            "Are there exercises for improving {}?",
        ],
        "adl_self_care": [
            "I can't {} due to limited mobility",
            "What tools help with {}?",
            "How can I {} more independently?",
            "I struggle with {}, what adaptations exist?",
            "Making {} easier with arthritis?",
        ],
        "senior_medication": [
            "How do I manage {} medications?",
            "I keep forgetting my {}, what can help?",
            "Is it safe to take {} with {}?",
            "How should I organize my {} medicines?",
            "What's the best way to remember {}?",
        ],
        "senior_social": [
            "My {} feels isolated, how can I help?",
            "Where can seniors meet {} in my area?",
            "How do I help my {} stay connected?",
            "What activities help with senior {}?",
            "Are there {} programs for elderly?",
        ],
        "mental_health_anxiety": [
            "I feel anxious about {}, what helps?",
            "How do I cope with {} anxiety?",
            "My anxiety about {} is overwhelming",
            "What techniques help with {} worry?",
            "I panic when {}, what should I do?",
        ],
        "mental_health_depression": [
            "I feel depressed about {}, how to cope?",
            "No motivation for {}, what helps?",
            "How do I deal with {} when depressed?",
            "Depression makes {} hard, any tips?",
            "Feeling hopeless about {}, what now?",
        ],
        "caregiver_respite": [
            "I need a break from caring for my {}",
            "Where can I find respite care for {}?",
            "How long can {} stay in respite care?",
            "What's the cost of {} respite services?",
            "How do I prepare {} for respite care?",
        ],
        "caregiver_burnout": [
            "I'm exhausted caring for my {}",
            "Feeling resentful about {}, is this normal?",
            "No time for {} due to caregiving",
            "How do I prevent {} burnout?",
            "I can't handle {} anymore, help?",
        ],
        "disability_equipment": [
            "What equipment helps with {}?",
            "How do I get {} covered by insurance?",
            "Are there alternatives to {}?",
            "How much does {} equipment cost?",
            "Where can I try {} before buying?",
        ],
        "disability_rights": [
            "Can my employer deny {} accommodation?",
            "What are my rights regarding {}?",
            "How do I request {} at work/school?",
            "Is {} discrimination illegal?",
            "Who enforces {} accessibility laws?",
        ],
    }

    # Variables to fill in templates
    fill_values = {
        "adl_mobility": [
            "bed",
            "wheelchair",
            "car",
            "toilet",
            "chair",
            "walking",
            "balance",
            "stairs",
            "transfers",
            "standing",
        ],
        "adl_self_care": [
            "button shirts",
            "tie shoes",
            "shower safely",
            "brush teeth",
            "cut food",
            "bathing",
            "dressing",
            "grooming",
            "eating",
            "toileting",
        ],
        "senior_medication": [
            "multiple",
            "morning pills",
            "blood pressure pills",
            "diabetes medication",
            "heart medicines",
            "daily pills",
            "prescriptions",
        ],
        "senior_social": [
            "mother",
            "father",
            "grandmother",
            "parent",
            "spouse",
            "people",
            "friends",
            "loneliness",
            "social",
            "volunteer",
        ],
        "mental_health_anxiety": [
            "work",
            "health",
            "future",
            "social situations",
            "finances",
            "driving",
            "public speaking",
            "constant",
            "nighttime",
            "specific",
        ],
        "mental_health_depression": [
            "loss",
            "work",
            "chronic illness",
            "anything",
            "daily tasks",
            "life changes",
            "relationships",
            "isolation",
            "everything",
            "the future",
        ],
        "caregiver_respite": [
            "mother",
            "spouse",
            "disabled child",
            "parent with dementia",
            "husband",
            "weekend",
            "overnight",
            "daily",
            "emergency",
            "vacation",
        ],
        "caregiver_burnout": [
            "mother",
            "spouse",
            "parent",
            "self-care",
            "exercise",
            "friends",
            "caregiver",
            "this situation",
            "caregiving duties",
            "everything",
        ],
        "disability_equipment": [
            "mobility",
            "communication",
            "wheelchair",
            "adaptive equipment",
            "assistive technology",
            "home modifications",
            "computer access",
            "daily living",
            "bathroom",
            "vehicle",
        ],
        "disability_rights": [
            "wheelchair",
            "work from home",
            "service animal",
            "accommodation",
            "accessible parking",
            "reasonable modifications",
            "sign language interpreter",
            "disability",
            "ADA",
            "housing",
        ],
    }

    # Generate data
    for category, responses in HEALTHCARE_RESPONSES.items():
        if category == "crisis_mental_health":
            # Special handling for crisis
            crisis_triggers = [
                "I want to hurt myself",
                "I'm thinking about suicide",
                "I don't want to live anymore",
                "I'm planning to end it all",
                "Life isn't worth living",
            ]
            for trigger in crisis_triggers:
                for response in responses:
                    training_data.append(
                        {
                            "question": trigger,
                            "response": response,
                            "category": category,
                        }
                    )
        else:
            # Regular categories
            templates = question_templates.get(category, [])
            values = fill_values.get(category, [])

            for template in templates:
                for value in values[:5]:  # Use first 5 values per template
                    if "{}" in template:
                        # Count number of {} placeholders
                        placeholders = template.count("{}")
                        if placeholders == 1:
                            question = template.format(value)
                        elif placeholders == 2:
                            # For templates with 2 placeholders, use different values
                            value2 = values[(values.index(value) + 1) % len(values)]
                            question = template.format(value, value2)
                        else:
                            question = template
                    else:
                        question = template

                    # Pair with different responses
                    response = random.choice(responses)
                    training_data.append(
                        {
                            "question": question,
                            "response": response,
                            "category": category,
                        }
                    )

    return training_data


def train_healthcare_model(training_data):
    """Train model and log to MLflow"""

    # Prepare data
    questions = [item["question"] for item in training_data]
    responses = [item["response"] for item in training_data]
    categories = [item["category"] for item in training_data]

    # Use categories as labels instead of individual responses
    unique_categories = list(set(categories))
    category_to_idx = {c: i for i, c in enumerate(unique_categories)}
    labels = [category_to_idx[c] for c in categories]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        questions, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Also split responses for evaluation
    responses_train, responses_test = train_test_split(
        responses, test_size=0.2, random_state=42, stratify=labels
    )

    # Start MLflow run
    experiment_name = "Healthcare_AI_Model_Training"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"healthcare_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Log parameters
        mlflow.log_param("model_type", "TfidfVectorizer + MultinomialNB")
        mlflow.log_param("total_samples", len(training_data))
        mlflow.log_param("unique_responses", len(set(responses)))
        mlflow.log_param("categories", len(set(categories)))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Create and train pipeline
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        )

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)

        # Log category-specific metrics
        category_accuracy = {}
        y_pred_arr = np.array(y_pred)
        y_test_arr = np.array(y_test)
        for i, cat in enumerate(unique_categories):
            cat_pred = (y_pred_arr == i).sum()
            cat_true = (y_test_arr == i).sum()
            if cat_true > 0:
                cat_correct = ((y_pred_arr == i) & (y_test_arr == i)).sum()
                cat_acc = cat_correct / cat_true
                mlflow.log_metric(f"accuracy_{cat}", cat_acc)
                category_accuracy[cat] = cat_acc

        # Save model and mappings
        model_data = {
            "pipeline": pipeline,
            "category_mapping": {i: c for c, i in category_to_idx.items()},
            "categories": unique_categories,
            "healthcare_responses": HEALTHCARE_RESPONSES,
            "training_date": datetime.now().isoformat(),
        }

        # Log model (skip MLflow artifact storage due to S3 permissions)
        # mlflow.sklearn.log_model(
        #     pipeline,
        #     "healthcare_model",
        #     registered_model_name="HealthcareResponseModel"
        # )

        # Save locally too
        os.makedirs("models/healthcare_ai", exist_ok=True)
        with open("models/healthcare_ai/model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        # Log artifacts (skip due to S3 permissions)
        # mlflow.log_artifact("models/healthcare_ai/model.pkl")

        # Create and log summary
        summary = f"""
Healthcare AI Model Training Summary
=====================================
Total Training Samples: {len(training_data)}
Unique Responses: {len(set(responses))}
Categories: {len(unique_categories)}

Model Performance:
- Overall Accuracy: {accuracy:.2%}
- Precision: {precision:.2%}
- Recall: {recall:.2%}
- F1 Score: {f1:.2%}

Category Performance:
{chr(10).join([f'- {cat}: {acc:.2%}' for cat, acc in category_accuracy.items()])}

Training completed in {training_time:.2f} seconds
        """

        with open("training_summary.txt", "w") as f:
            f.write(summary)
        # mlflow.log_artifact("training_summary.txt")

        print(summary)
        print(f"\nModel trained and logged to MLflow!")
        print(f"View at: {MLFLOW_TRACKING_URI}")

        return model_data


if __name__ == "__main__":
    print("Generating diverse healthcare training data...")
    training_data = generate_training_data()
    print(f"Generated {len(training_data)} training samples")

    print("\nTraining healthcare model...")
    model_data = train_healthcare_model(training_data)

    print("\nModel training complete!")
    print(f"Model saved to: models/healthcare_ai/model.pkl")
