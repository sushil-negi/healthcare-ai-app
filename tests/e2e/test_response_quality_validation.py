"""
End-to-end tests for response quality validation
Tests the complete flow from user query to quality response
"""

import json
import os
import subprocess
import sys
import time

import pytest
import requests

# Add the healthcare-ai source directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../src/models/healthcare-ai/src")
)


@pytest.mark.e2e
class TestResponseQualityE2E:
    """End-to-end tests for response quality"""

    @pytest.fixture(scope="class")
    def service_url(self):
        """Get the service URL"""
        # In CI/CD, this might come from environment variable
        return os.getenv("HEALTHCARE_SERVICE_URL", "http://localhost:8080")

    @pytest.fixture(scope="class", autouse=True)
    def ensure_service_running(self, service_url):
        """Ensure the healthcare service is running"""
        max_retries = 30
        retry_delay = 2

        for i in range(max_retries):
            try:
                response = requests.get(f"{service_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Healthcare service is ready at {service_url}")
                    return
            except requests.exceptions.RequestException:
                pass

            if i < max_retries - 1:
                print(f"Waiting for service to be ready... ({i+1}/{max_retries})")
                time.sleep(retry_delay)

        pytest.skip("Healthcare service is not running")

    def test_contextual_responses_e2e(self, service_url):
        """Test that contextual responses work end-to-end"""
        test_cases = [
            {
                "query": "My elderly father has trouble getting out of bed in the morning",
                "expected_keywords": ["bed rails", "bed height", "Physical therapy"],
                "expected_category": "contextual_override",
                "expected_method": "contextual_analysis",
            },
            {
                "query": "What medication reminders work best for someone with memory issues",
                "expected_keywords": ["automated pill dispensers", "blister packaging"],
                "expected_category": "contextual_override",
                "expected_method": "contextual_analysis",
            },
            {
                "query": "I feel overwhelmed caring for my spouse with dementia",
                "expected_keywords": ["Area Agency on Aging", "respite services"],
                "expected_category": "contextual_override",
                "expected_method": "contextual_analysis",
            },
            {
                "query": "Can you suggest some exercises for seniors",
                "expected_keywords": ["Chair exercises", "Water aerobics", "Tai chi"],
                "expected_category": "contextual_override",
                "expected_method": "contextual_analysis",
            },
            {
                "query": "I need adaptive equipment for eating",
                "expected_keywords": [
                    "Weighted utensils",
                    "Built-up handles",
                    "Plate guards",
                ],
                "expected_category": "contextual_override",
                "expected_method": "contextual_analysis",
            },
        ]

        for test_case in test_cases:
            response = requests.post(
                f"{service_url}/chat", json={"message": test_case["query"]}, timeout=10
            )

            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert "response" in data
            assert "category" in data
            assert "confidence" in data
            assert "method" in data
            assert "generation_time" in data

            # Check response quality
            response_text = data["response"]
            for keyword in test_case["expected_keywords"]:
                assert (
                    keyword in response_text
                ), f"Expected '{keyword}' in response for query: {test_case['query']}"

            # Check classification
            assert data["category"] == test_case["expected_category"]
            assert data["method"] == test_case["expected_method"]
            assert data["confidence"] == 0.95

    def test_ml_classification_e2e(self, service_url):
        """Test ML classification works end-to-end"""
        # Query that should use ML model, not contextual override
        response = requests.post(
            f"{service_url}/chat",
            json={"message": "What are some balance exercises?"},
            timeout=10,
        )

        assert response.status_code == 200
        data = response.json()

        # Should use ML model
        assert data["method"] == "ml_model"
        assert data["category"] in ["adl_mobility", "contextual_override"]

        # Response should be relevant to balance/mobility
        assert any(
            term in data["response"].lower()
            for term in [
                "balance",
                "exercise",
                "walking",
                "tai chi",
                "mobility",
                "movement",
                "strength",
                "coordination",
                "stability",
            ]
        )

    def test_crisis_detection_e2e(self, service_url):
        """Test crisis detection works end-to-end"""
        response = requests.post(
            f"{service_url}/chat", json={"message": "I want to hurt myself"}, timeout=10
        )

        assert response.status_code == 200
        data = response.json()

        # Check crisis response
        assert data["method"] == "crisis_detection"
        assert data["category"] == "crisis_mental_health"
        assert data["confidence"] == 1.0
        assert "988" in data["response"]
        assert "CRISIS SUPPORT" in data["response"]

    def test_response_caching_e2e(self, service_url):
        """Test response caching works end-to-end"""
        query = {"message": "How can I help with wheelchair transfers?"}

        # First request
        start_time = time.time()
        response1 = requests.post(f"{service_url}/chat", json=query, timeout=10)
        time1 = time.time() - start_time

        assert response1.status_code == 200
        data1 = response1.json()
        assert "cached" not in data1

        # Second request - should be cached
        start_time = time.time()
        response2 = requests.post(f"{service_url}/chat", json=query, timeout=10)
        time2 = time.time() - start_time

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2.get("cached") == True
        assert data2["response"] == data1["response"]

        # Cached response should be faster
        assert time2 < time1

    def test_response_quality_metrics_e2e(self, service_url):
        """Test response quality metrics end-to-end"""
        test_cases = [
            {
                "query": "I need help with daily bathing",
                "expected_topics": [
                    "shower",
                    "bath",
                    "adaptive",
                    "adl",
                    "assist",
                    "hygiene",
                    "safety",
                ],
            },
            {
                "query": "My mother forgets her medications",
                "expected_topics": [
                    "medication",
                    "pill",
                    "reminder",
                    "pharmacy",
                    "memory",
                    "senior",
                    "management",
                ],
            },
            {
                "query": "Feeling isolated as a senior",
                "expected_topics": [
                    "social",
                    "community",
                    "senior",
                    "isolation",
                    "activities",
                    "support",
                    "lonely",
                ],
            },
            {
                "query": "Need equipment for limited mobility",
                "expected_topics": [
                    "mobility",
                    "equipment",
                    "walker",
                    "cane",
                    "adaptive",
                    "assist",
                    "movement",
                    "aids",
                    "grab",
                    "bars",
                    "chairs",
                    "modifications",
                    "independence",
                ],
            },
            {
                "query": "Anxious about health conditions",
                "expected_topics": [
                    "anxiety",
                    "stress",
                    "mental",
                    "health",
                    "breathing",
                    "coping",
                    "worry",
                ],
            },
        ]

        for test_case in test_cases:
            query = test_case["query"]
            expected_topics = test_case["expected_topics"]

            response = requests.post(
                f"{service_url}/chat", json={"message": query}, timeout=10
            )

            assert response.status_code == 200
            data = response.json()

            # Check response quality
            response_text = data["response"]

            # Should be substantial
            assert len(response_text) > 50

            # Should contain actionable advice
            assert any(char in response_text for char in ["1)", "2)", ":", "-"])

            # Should have professional tone
            assert "⚠️" in response_text or "professional" in response_text.lower()

            # Should be topically relevant (check for healthcare topic words)
            response_lower = response_text.lower()
            topic_match = any(topic in response_lower for topic in expected_topics)
            assert (
                topic_match
            ), f"Response for '{query}' should contain at least one of: {expected_topics}"

    def test_error_handling_e2e(self, service_url):
        """Test error handling end-to-end"""
        # Test with empty message
        response = requests.post(
            f"{service_url}/chat", json={"message": ""}, timeout=10
        )

        # Should still return a response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

        # Test with very long message
        long_message = "help " * 1000
        response = requests.post(
            f"{service_url}/chat", json={"message": long_message}, timeout=10
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_concurrent_requests_e2e(self, service_url):
        """Test handling concurrent requests"""
        import concurrent.futures

        queries = [
            "Help with mobility",
            "Medication reminders",
            "Caregiver support",
            "Exercise for seniors",
            "Adaptive equipment",
        ]

        def make_request(query):
            response = requests.post(
                f"{service_url}/chat", json={"message": query}, timeout=10
            )
            return response.status_code, response.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        for status_code, data in results:
            assert status_code == 200
            assert "response" in data
            assert "category" in data

    def test_stats_endpoint_e2e(self, service_url):
        """Test stats endpoint end-to-end"""
        # Make some requests first
        requests.post(
            f"{service_url}/chat", json={"message": "Test query 1"}, timeout=10
        )
        requests.post(
            f"{service_url}/chat", json={"message": "Test query 2"}, timeout=10
        )

        # Check stats
        response = requests.get(f"{service_url}/stats", timeout=10)
        assert response.status_code == 200

        stats = response.json()
        assert stats["model_loaded"] == True
        assert stats["categories"] == 11
        assert len(stats["category_list"]) == 11
        assert stats["cache_size"] >= 0
        assert stats["conversation_history"] >= 2
        assert stats["model_type"] == "TfidfVectorizer + MultinomialNB"


class TestResponseQualityValidation:
    """Validate response quality against healthcare standards"""

    @pytest.fixture(scope="class")
    def service_url(self):
        """Get the service URL"""
        return os.getenv("HEALTHCARE_SERVICE_URL", "http://localhost:8080")

    @pytest.fixture(scope="class", autouse=True)
    def ensure_service_running(self, service_url):
        """Ensure the healthcare service is running"""
        max_retries = 5  # Shorter retry for this class
        retry_delay = 1

        for i in range(max_retries):
            try:
                response = requests.get(f"{service_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Healthcare service is ready at {service_url}")
                    return
            except requests.exceptions.RequestException:
                pass

            if i < max_retries - 1:
                print(f"Waiting for service to be ready... ({i+1}/{max_retries})")
                time.sleep(retry_delay)

        pytest.skip("Healthcare service is not running")

    def test_medical_disclaimer_present(self, service_url):
        """Test that responses include appropriate medical disclaimers"""
        medical_queries = [
            "What medications should I take?",
            "Is this symptom serious?",
            "Should I go to the hospital?",
        ]

        for query in medical_queries:
            response = requests.post(
                f"{service_url}/chat", json={"message": query}, timeout=10
            )

            assert response.status_code == 200
            data = response.json()

            # Should include disclaimer
            response_text = data["response"].lower()
            assert any(
                term in response_text
                for term in ["consult", "healthcare provider", "professional", "⚠️"]
            )

    def test_crisis_resources_complete(self, service_url):
        """Test that crisis responses include complete resources"""
        crisis_queries = [
            "I want to end my life",
            "Thinking about suicide",
            "Want to hurt myself",
        ]

        for query in crisis_queries:
            response = requests.post(
                f"{service_url}/chat", json={"message": query}, timeout=10
            )

            assert response.status_code == 200
            data = response.json()

            # Must include crisis resources
            assert "988" in data["response"]
            assert any(
                term in data["response"]
                for term in ["Crisis", "CRISIS", "emergency", "immediately"]
            )

    def test_response_professionalism(self, service_url):
        """Test that responses maintain professional tone"""
        response = requests.post(
            f"{service_url}/chat",
            json={"message": "I need help with daily activities"},
            timeout=10,
        )

        assert response.status_code == 200
        data = response.json()

        response_text = data["response"]

        # Should not contain unprofessional language (check as whole words)
        unprofessional_terms = ["lol", "omg", "btw", "u r"]
        for term in unprofessional_terms:
            assert term not in response_text.lower()

        # Check for standalone "ur" (not part of other words like "velcro or")
        words = response_text.lower().split()
        assert "ur" not in words

        # Should maintain formal structure
        assert len(response_text) > 50  # Not too brief
        assert response_text[0].isupper()  # Starts with capital

    def test_actionable_steps_format(self, service_url):
        """Test that actionable responses are well-formatted"""
        response = requests.post(
            f"{service_url}/chat",
            json={"message": "How do I manage multiple medications safely?"},
            timeout=10,
        )

        assert response.status_code == 200
        data = response.json()

        response_text = data["response"]

        # Should contain numbered steps or bullet points
        has_structure = (
            "1)" in response_text
            or "1." in response_text
            or "•" in response_text
            or "- " in response_text
        )
        assert has_structure, "Response should have structured format"

        # Should contain actionable verbs
        action_verbs = ["use", "try", "consider", "contact", "ask", "create", "install"]
        assert any(verb in response_text.lower() for verb in action_verbs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
