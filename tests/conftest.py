"""
Shared pytest fixtures for all tests
"""

import os
import pickle
import sys

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Add the healthcare-ai source directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../src/models/healthcare-ai/src")
)

from healthcare_trained_engine import HealthcareTrainedEngine


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """Create a trained model for testing - shared across all test classes"""
    # Sample training data covering all categories
    training_data = [
        # ADL Mobility
        ("I need help transferring from bed to wheelchair", "adl_mobility"),
        ("What exercises can improve my balance?", "adl_mobility"),
        ("How do I safely use a walker?", "adl_mobility"),
        ("My legs are weak, what mobility aids help?", "adl_mobility"),
        # ADL Self-care
        ("I struggle with dressing due to arthritis", "adl_self_care"),
        ("What tools help with bathing independently?", "adl_self_care"),
        ("Eating is difficult with tremors", "adl_self_care"),
        ("Need help with grooming tasks", "adl_self_care"),
        # Senior Medication
        ("How do I manage multiple medications?", "senior_medication"),
        ("I keep forgetting to take my pills", "senior_medication"),
        ("What's the best pill organizer for seniors?", "senior_medication"),
        ("Medication side effects concern me", "senior_medication"),
        # Senior Social
        ("My mother feels isolated and lonely", "senior_social"),
        ("Where can seniors meet friends?", "senior_social"),
        ("How to stay socially connected as elderly?", "senior_social"),
        ("Senior activity programs near me", "senior_social"),
        # Mental Health Anxiety
        ("I'm anxious about my health condition", "mental_health_anxiety"),
        ("Panic attacks are affecting my life", "mental_health_anxiety"),
        ("How to manage anxiety as a senior?", "mental_health_anxiety"),
        ("Worried constantly about everything", "mental_health_anxiety"),
        # Mental Health Depression
        ("Feeling depressed and hopeless", "mental_health_depression"),
        ("No motivation to do anything anymore", "mental_health_depression"),
        ("Depression makes daily tasks hard", "mental_health_depression"),
        ("Sad all the time, need help", "mental_health_depression"),
        # Caregiver Respite
        ("I need a break from caregiving", "caregiver_respite"),
        ("Where can I find respite care?", "caregiver_respite"),
        ("How much does respite care cost?", "caregiver_respite"),
        ("Temporary care for my spouse needed", "caregiver_respite"),
        # Caregiver Burnout
        ("I'm exhausted from caring for my parent", "caregiver_burnout"),
        ("Feeling resentful about caregiving", "caregiver_burnout"),
        ("No time for myself anymore", "caregiver_burnout"),
        ("Caregiver stress is overwhelming", "caregiver_burnout"),
        # Disability Equipment
        ("What wheelchair is best for me?", "disability_equipment"),
        ("Need communication devices for speech", "disability_equipment"),
        ("Home modifications for wheelchair access", "disability_equipment"),
        ("Adaptive computer equipment needed", "disability_equipment"),
        # Disability Rights
        ("My employer won't provide accommodations", "disability_rights"),
        ("ADA rights for wheelchair users", "disability_rights"),
        ("Housing discrimination due to disability", "disability_rights"),
        ("Educational accommodations for disability", "disability_rights"),
        # Crisis Mental Health
        ("I want to end my life", "crisis_mental_health"),
        ("Thinking about hurting myself", "crisis_mental_health"),
        ("Life isn't worth living anymore", "crisis_mental_health"),
        ("Suicidal thoughts won't stop", "crisis_mental_health"),
    ]

    # Prepare data
    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]

    # Create category mapping
    unique_categories = sorted(list(set(labels)))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}

    # Convert labels to indices
    label_indices = [category_to_idx[label] for label in labels]

    # Train model
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )

    pipeline.fit(texts, label_indices)

    # Create dummy responses for testing
    healthcare_responses = {
        cat: [f"Test response for {cat}"] for cat in unique_categories
    }

    # Save model
    model_data = {
        "pipeline": pipeline,
        "category_mapping": idx_to_category,
        "healthcare_responses": healthcare_responses,
    }

    model_path = tmp_path_factory.mktemp("models") / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    return model_path
