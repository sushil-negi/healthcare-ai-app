#!/usr/bin/env python3
"""
Mock healthcare model class for CI testing
"""

import numpy as np


class MockHealthcareModel:
    """Mock healthcare AI model for testing"""
    
    def __init__(self):
        self.model_type = "mock_healthcare_model"
        self.version = "1.0.0"
        self.categories = [
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
            "general_healthcare"
        ]
        self.crisis_keywords = [
            "suicide",
            "kill myself", 
            "end it all",
            "hurt myself",
            "die",
        ]

    def predict(self, X):
        """Mock predict method that returns reasonable categories"""
        if isinstance(X, str):
            X = [X]
        
        predictions = []
        for query in X:
            query_lower = query.lower()
            
            # Crisis detection
            if any(keyword in query_lower for keyword in self.crisis_keywords):
                predictions.append("crisis_mental_health")
            # Mobility/ADL
            elif any(word in query_lower for word in ["bed", "mobility", "walking", "balance"]):
                predictions.append("adl_mobility")
            # Medication
            elif any(word in query_lower for word in ["medication", "pill", "drug"]):
                predictions.append("senior_medication")
            # Mental health
            elif any(word in query_lower for word in ["anxious", "depressed", "worry"]):
                predictions.append("mental_health_anxiety")
            # Default
            else:
                predictions.append("general_healthcare")
                
        return predictions

    def predict_proba(self, X):
        """Mock predict_proba method"""
        if isinstance(X, str):
            X = [X]
            
        proba_matrix = []
        for _ in X:
            # Create random probabilities that sum to 1
            probs = np.random.dirichlet(np.ones(len(self.categories)))
            proba_matrix.append(probs)
            
        return np.array(proba_matrix)