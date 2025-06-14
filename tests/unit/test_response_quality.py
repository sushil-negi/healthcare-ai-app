"""
Unit tests for Healthcare AI response quality, classification accuracy, and contextual overrides
"""

import os
import pickle
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the healthcare-ai source directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../src/models/healthcare-ai/src")
)

from healthcare_trained_engine import HealthcareTrainedEngine


class TestResponseQuality:
    """Test response quality and specificity"""

    @pytest.fixture
    def mock_model_data(self):
        """Create mock model data for testing"""
        # Create a real pipeline instead of MagicMock to avoid pickling issues
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        # Train with minimal data
        X = ["mobility help", "medication help", "social help"]
        y = [0, 2, 3]
        pipeline.fit(X, y)

        return {
            "pipeline": pipeline,
            "category_mapping": {
                0: "adl_mobility",
                1: "adl_self_care",
                2: "senior_medication",
                3: "senior_social",
                4: "mental_health_anxiety",
                5: "mental_health_depression",
                6: "caregiver_respite",
                7: "caregiver_burnout",
                8: "disability_equipment",
                9: "disability_rights",
                10: "crisis_mental_health",
            },
            "healthcare_responses": {
                "adl_mobility": [
                    "Balance exercises that help: heel-to-toe walking, standing on one foot (near support), gentle tai chi movements, seated leg lifts, and ankle pumps. Start with 5-10 minutes daily, always near sturdy support.",
                    "For wheelchair transfers, the key techniques include: 1) Lock the wheelchair brakes, 2) Position at 45-degree angle to target surface, 3) Use sliding board if needed, 4) Keep back straight and pivot with legs. Would you like specific guidance for bed, car, or toilet transfers?",
                ],
                "senior_social": [
                    "Combating senior isolation: join senior centers for activities and meals, participate in faith communities, volunteer at libraries or schools, join hobby clubs, or attend community education classes."
                ],
                "senior_medication": [
                    "Medication management systems: weekly pill organizers with AM/PM compartments, automatic pill dispensers with alarms, medication reminder apps, and pharmacy blister packs. Consider large-print labels."
                ],
                "caregiver_respite": [
                    "Planning respite breaks: start with short breaks (2-4 hours), gradually increase duration, prepare care recipient for changes, leave detailed care instructions, and use respite time for self-care not just errands."
                ],
            },
        }

    @pytest.fixture
    def engine_with_mock_model(self, mock_model_data, tmp_path):
        """Create engine with mocked model"""
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(mock_model_data, f)

        engine = HealthcareTrainedEngine(str(model_path))
        return engine

    def test_contextual_override_bed_mobility(self, engine_with_mock_model):
        """Test contextual override for elderly bed mobility"""
        response = engine_with_mock_model.generate_response(
            "My elderly father has trouble getting out of bed in the morning"
        )

        assert response["method"] == "contextual_analysis"
        assert response["category"] == "contextual_override"
        assert response["confidence"] == 0.95
        assert "bed rails" in response["response"]
        assert "bed height" in response["response"]
        assert "Physical therapy" in response["response"]

    def test_contextual_override_medication_memory(self, engine_with_mock_model):
        """Test contextual override for medication with memory issues"""
        response = engine_with_mock_model.generate_response(
            "What medication reminders work best for someone with memory issues"
        )

        assert response["method"] == "contextual_analysis"
        assert response["category"] == "contextual_override"
        assert "automated pill dispensers" in response["response"]
        assert "blister packaging" in response["response"]
        assert "medication synchronization" in response["response"]

    def test_contextual_override_caregiver_overwhelm(self, engine_with_mock_model):
        """Test contextual override for caregiver overwhelm"""
        response = engine_with_mock_model.generate_response(
            "I feel overwhelmed caring for my spouse with dementia"
        )

        assert response["method"] == "contextual_analysis"
        assert response["category"] == "contextual_override"
        assert "Area Agency on Aging" in response["response"]
        assert "respite services" in response["response"]
        assert "Your wellbeing directly impacts care quality" in response["response"]

    def test_contextual_override_senior_exercises(self, engine_with_mock_model):
        """Test contextual override for senior exercises"""
        response = engine_with_mock_model.generate_response(
            "Can you suggest some exercises for seniors"
        )

        assert response["method"] == "contextual_analysis"
        assert response["category"] == "contextual_override"
        assert "Chair exercises" in response["response"]
        assert "Water aerobics" in response["response"]
        assert "Tai chi" in response["response"]
        assert "Start slowly" in response["response"]

    def test_contextual_override_eating_equipment(self, engine_with_mock_model):
        """Test contextual override for eating equipment"""
        response = engine_with_mock_model.generate_response(
            "I need adaptive equipment for eating"
        )

        assert response["method"] == "contextual_analysis"
        assert response["category"] == "contextual_override"
        assert "Weighted utensils" in response["response"]
        assert "Built-up handles" in response["response"]
        assert "Plate guards" in response["response"]
        assert "dining independence" in response["response"]

    def test_contextual_override_wheelchair_transfers(self, engine_with_mock_model):
        """Test contextual override for wheelchair transfers"""
        response = engine_with_mock_model.generate_response(
            "How do I help my mother with wheelchair transfers to her car"
        )

        assert response["method"] == "contextual_analysis"
        assert "Lock wheelchair brakes" in response["response"]
        assert "45° angle" in response["response"]
        assert "transfer board" in response["response"]

    def test_contextual_override_depression(self, engine_with_mock_model):
        """Test contextual override for depression"""
        response = engine_with_mock_model.generate_response(
            "I have been feeling depressed lately and need help"
        )

        assert response["method"] == "contextual_analysis"
        assert "Depression affects many people and is treatable" in response["response"]
        assert "988" in response["response"]
        assert "mental health professional" in response["response"]

    def test_contextual_override_anxiety(self, engine_with_mock_model):
        """Test contextual override for anxiety"""
        response = engine_with_mock_model.generate_response(
            "I'm feeling very anxious about my health"
        )

        assert response["method"] == "contextual_analysis"
        assert "deep breathing" in response["response"]
        assert "grounding techniques" in response["response"]
        assert "professional help" in response["response"]


class TestMLClassificationAccuracy:
    """Test ML model classification accuracy"""

    @pytest.fixture
    def engine_with_predictions(self, tmp_path):
        """Create engine with predictable ML model"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        # Create and train a real pipeline
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        # Train with data for all categories
        X = [
            "mobility help",
            "self care",
            "medication",
            "social",
            "anxiety",
            "depression",
            "respite",
            "burnout",
            "equipment",
            "rights",
            "crisis",
        ]
        y = list(range(11))
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {
                i: cat
                for i, cat in enumerate(
                    [
                        "adl_mobility",
                        "adl_self_care",
                        "senior_medication",
                        "senior_social",
                        "mental_health_anxiety",
                        "mental_health_depression",
                        "caregiver_respite",
                        "caregiver_burnout",
                        "disability_equipment",
                        "disability_rights",
                        "crisis_mental_health",
                    ]
                )
            },
            "healthcare_responses": {
                "adl_mobility": ["Balance exercises that help..."],
                "senior_medication": ["Medication management systems..."],
            },
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        engine = HealthcareTrainedEngine(str(model_path))
        return engine, pipeline

    def test_ml_classification_mobility(self, engine_with_predictions):
        """Test ML classification for mobility queries"""
        engine, pipeline = engine_with_predictions

        response = engine.generate_response("I need help with mobility and walking")

        # Should classify as mobility or use contextual override
        assert response["category"] in ["adl_mobility", "contextual_override"]
        assert response["method"] in ["ml_model", "contextual_analysis"]

    def test_ml_classification_medication(self, engine_with_predictions):
        """Test ML classification for medication queries"""
        engine, pipeline = engine_with_predictions

        response = engine.generate_response(
            "How should I organize my medications and pills"
        )

        # Should classify as medication or use contextual override
        assert response["category"] in ["senior_medication", "contextual_override"]
        assert response["method"] in ["ml_model", "contextual_analysis"]

    def test_ml_classification_confidence_threshold(self, engine_with_predictions):
        """Test ML classification with various confidence levels"""
        engine, pipeline = engine_with_predictions

        response = engine.generate_response(
            "General healthcare question about wellness"
        )

        # Should have some confidence score
        assert 0.0 <= response["confidence"] <= 1.0
        assert response["method"] in ["ml_model", "contextual_analysis"]


class TestResponseSelection:
    """Test contextual response selection logic"""

    def test_exercise_keyword_scoring(self):
        """Test exercise-specific keyword scoring"""
        engine = HealthcareTrainedEngine.__new__(HealthcareTrainedEngine)
        engine.healthcare_responses = {
            "adl_mobility": [
                "General mobility advice",
                "Balance exercises that help: heel-to-toe walking, tai chi",
                "Walking tips for seniors",
            ]
        }

        response = engine._select_contextual_response(
            "What exercises can help with balance",
            engine.healthcare_responses["adl_mobility"],
        )

        assert "Balance exercises" in response
        assert "tai chi" in response

    def test_eating_keyword_scoring(self):
        """Test eating-specific keyword scoring"""
        engine = HealthcareTrainedEngine.__new__(HealthcareTrainedEngine)
        engine.healthcare_responses = {
            "adl_self_care": [
                "General self-care tips",
                "Eating adaptations: weighted utensils, plate guards",
                "Bathing safety tips",
            ]
        }

        response = engine._select_contextual_response(
            "I need help with eating utensils",
            engine.healthcare_responses["adl_self_care"],
        )

        assert "Eating adaptations" in response
        assert "weighted utensils" in response


class TestCrisisDetection:
    """Test crisis detection and response"""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine for crisis testing"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        X = ["test"]
        y = [0]
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {0: "crisis_mental_health"},
            "healthcare_responses": {
                "crisis_mental_health": [
                    "🚨 CRISIS SUPPORT 🚨\n\nCall 988 or 911 immediately."
                ]
            },
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return HealthcareTrainedEngine(str(model_path))

    def test_crisis_detection_suicide(self, engine):
        """Test crisis detection for suicide mentions"""
        response = engine.generate_response("I want to kill myself")

        assert response["category"] == "crisis_mental_health"
        assert response["method"] == "crisis_detection"
        assert response["confidence"] == 1.0
        assert "988" in response["response"]
        assert "CRISIS SUPPORT" in response["response"]

    def test_crisis_detection_self_harm(self, engine):
        """Test crisis detection for self-harm mentions"""
        response = engine.generate_response("I want to hurt myself")

        assert response["category"] == "crisis_mental_health"
        assert response["method"] == "crisis_detection"
        assert "988" in response["response"]

    def test_non_crisis_with_sensitive_words(self, engine):
        """Test that non-crisis contexts don't trigger crisis response"""
        response = engine.generate_response(
            "How can I prevent my teenager from thoughts of self-harm"
        )

        # Should not trigger crisis response for prevention questions
        assert response["method"] != "crisis_detection"


class TestResponseCaching:
    """Test response caching functionality"""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine for caching tests"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        X = ["test"]
        y = [0]
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {0: "adl_mobility"},
            "healthcare_responses": {"adl_mobility": ["Test response for caching"]},
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return HealthcareTrainedEngine(str(model_path))

    def test_response_caching(self, engine):
        """Test that responses are cached"""
        query = "How can I improve my balance"

        # First call
        response1 = engine.generate_response(query)
        assert "cached" not in response1

        # Second call - should be cached
        response2 = engine.generate_response(query)
        assert response2.get("cached") == True
        assert response1["response"] == response2["response"]

    def test_cache_case_insensitive(self, engine):
        """Test that cache is case-insensitive"""
        response1 = engine.generate_response("How can I improve my balance")
        response2 = engine.generate_response("HOW CAN I IMPROVE MY BALANCE")

        assert response2.get("cached") == True


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_missing_model_file(self):
        """Test handling of missing model file"""
        with pytest.raises(FileNotFoundError):
            engine = HealthcareTrainedEngine("/nonexistent/path/model.pkl")

    def test_ml_prediction_error(self, tmp_path):
        """Test handling of ML prediction errors"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        # Create a working pipeline first
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        X = ["test"]
        y = [0]
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {},  # Empty mapping will cause error
            "healthcare_responses": {},
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        engine = HealthcareTrainedEngine(str(model_path))
        response = engine.generate_response("Test query")

        assert response["category"] == "error"
        assert response["method"] == "error"
        assert response["confidence"] == 0.0
        assert "difficulty processing" in response["response"]


class TestResponseFormatting:
    """Test response formatting and structure"""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine for formatting tests"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        X = ["test"]
        y = [0]
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {0: "adl_mobility"},
            "healthcare_responses": {
                "adl_mobility": [
                    "Test response for formatting. ⚠️ Consult healthcare providers."
                ]
            },
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return HealthcareTrainedEngine(str(model_path))

    def test_response_structure(self, engine):
        """Test that all responses have required fields"""
        response = engine.generate_response("I need help with mobility")

        required_fields = [
            "response",
            "category",
            "confidence",
            "method",
            "generation_time",
        ]
        for field in required_fields:
            assert field in response

        assert isinstance(response["response"], str)
        assert isinstance(response["category"], str)
        assert isinstance(response["confidence"], float)
        assert isinstance(response["method"], str)
        assert isinstance(response["generation_time"], float)

    def test_response_contains_warning(self, engine):
        """Test that responses contain appropriate warnings"""
        response = engine.generate_response(
            "My elderly father has trouble getting out of bed"
        )

        assert "⚠️" in response["response"]
        assert "Consult healthcare providers" in response["response"]

    def test_response_numbered_steps(self, engine):
        """Test that contextual responses have numbered steps"""
        response = engine.generate_response("I need adaptive equipment for eating")

        assert "1)" in response["response"]
        assert "2)" in response["response"]
        assert "3)" in response["response"]


class TestConversationHistory:
    """Test conversation history tracking"""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine for history tests"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        X = ["test"]
        y = [0]
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {0: "adl_mobility"},
            "healthcare_responses": {"adl_mobility": ["Test response"]},
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return HealthcareTrainedEngine(str(model_path))

    def test_conversation_history_tracking(self, engine):
        """Test that conversation history is tracked"""
        initial_history_length = len(engine.conversation_history)

        engine.generate_response("First question")
        engine.generate_response("Second question")

        assert len(engine.conversation_history) == initial_history_length + 2

        # Check history structure
        last_entry = engine.conversation_history[-1]
        assert "user" in last_entry
        assert "assistant" in last_entry
        assert "category" in last_entry
        assert "confidence" in last_entry
        assert "timestamp" in last_entry


class TestEngineStatistics:
    """Test engine statistics functionality"""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine for stats tests"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("classifier", MultinomialNB()),
            ]
        )

        X = ["test"] * 11
        y = list(range(11))
        pipeline.fit(X, y)

        model_data = {
            "pipeline": pipeline,
            "category_mapping": {i: f"cat_{i}" for i in range(11)},
            "healthcare_responses": {f"cat_{i}": ["Response"] for i in range(11)},
        }

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return HealthcareTrainedEngine(str(model_path))

    def test_get_stats(self, engine):
        """Test engine statistics"""
        stats = engine.get_stats()

        assert stats["model_loaded"] == True
        assert stats["categories"] == 11
        assert len(stats["category_list"]) == 11
        assert stats["model_type"] == "TfidfVectorizer + MultinomialNB"
        assert isinstance(stats["cache_size"], int)
        assert isinstance(stats["conversation_history"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
