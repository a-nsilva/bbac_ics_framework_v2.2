#!/usr/bin/env python3
"""
BBAC ICS Framework - Statistical Analysis Models

Statistical methods for behavioral analysis:
- Action frequency deviation
- Resource usage patterns
- Temporal analysis
- Location distribution
- Request frequency analysis
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..util.data_structures import (
    AccessRequest,
    LayerDecision,
    DecisionType,
)


logger = logging.getLogger(__name__)


class StatisticalModel:
    """
    Statistical analysis model for behavioral patterns.
    
    Analyzes requests against baseline to detect deviations.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5
    ):
        """
        Initialize statistical model.
        
        Args:
            weights: Feature weights for scoring
            threshold: Decision threshold (score >= threshold → grant)
        """
        if weights is None:
            # Default weights (sum to 1.0)
            weights = {
                'action_deviation': 0.25,
                'resource_deviation': 0.25,
                'temporal_deviation': 0.20,
                'location_deviation': 0.15,
                'frequency_deviation': 0.15,
            }
        
        self.weights = weights
        self.threshold = threshold
        
        # Validate weights
        total = sum(weights.values())
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total:.3f}, not 1.0")
        
        logger.info(
            f"StatisticalModel initialized: threshold={threshold}, "
            f"weights={weights}"
        )
    
    def analyze(
        self,
        request: AccessRequest,
        baseline: Optional[Dict],
        features: Optional[Dict] = None
    ) -> LayerDecision:
        """
        Analyze request using statistical methods.
        
        Args:
            request: Access request
            baseline: Agent baseline profile
            features: Pre-extracted features (optional)
            
        Returns:
            LayerDecision with statistical analysis result
        """
        import time
        start_time = time.time()
        
        # If no baseline, grant with low confidence
        if baseline is None or not baseline:
            return LayerDecision(
                decision=DecisionType.GRANT,
                confidence=0.5,
                latency_ms=(time.time() - start_time) * 1000,
                layer_name='statistical',
                explanation={'reason': 'no_baseline', 'score': 0.5}
            )
        
        # Calculate individual scores
        scores = {}
        
        # 1. Action deviation (25%)
        scores['action_deviation'] = self._calculate_action_deviation(
            request, baseline
        )
        
        # 2. Resource deviation (25%)
        scores['resource_deviation'] = self._calculate_resource_deviation(
            request, baseline
        )
        
        # 3. Temporal deviation (20%)
        scores['temporal_deviation'] = self._calculate_temporal_deviation(
            request, baseline, features
        )
        
        # 4. Location deviation (15%)
        scores['location_deviation'] = self._calculate_location_deviation(
            request, baseline
        )
        
        # 5. Frequency deviation (15%)
        scores['frequency_deviation'] = self._calculate_frequency_deviation(
            request, baseline, features
        )
        
        # Weighted aggregation
        aggregate_score = sum(
            scores[key] * self.weights[key]
            for key in scores.keys()
        )
        
        # Decision
        decision = (
            DecisionType.GRANT if aggregate_score >= self.threshold
            else DecisionType.DENY
        )
        
        # Confidence based on score distance from threshold
        if decision == DecisionType.GRANT:
            confidence = 0.5 + (aggregate_score - self.threshold) / (1.0 - self.threshold) * 0.5
        else:
            confidence = 0.5 + (self.threshold - aggregate_score) / self.threshold * 0.5
        
        confidence = np.clip(confidence, 0.0, 1.0)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LayerDecision(
            decision=decision,
            confidence=float(confidence),
            latency_ms=float(latency_ms),
            layer_name='statistical',
            explanation={
                'aggregate_score': float(aggregate_score),
                'threshold': self.threshold,
                'scores': {k: float(v) for k, v in scores.items()},
            }
        )
    
    def _calculate_action_deviation(
        self,
        request: AccessRequest,
        baseline: Dict
    ) -> float:
        """
        Calculate action frequency deviation.
        
        Returns score in [0, 1] where 1 = normal, 0 = very abnormal.
        """
        action = request.action.value
        
        # Get action frequencies from baseline
        common_actions = baseline.get('common_actions', {})
        frequencies = common_actions.get('frequencies', {})
        
        if not frequencies:
            return 0.5  # Neutral if no data
        
        # Get frequency for this action
        action_freq = frequencies.get(action, 0.0)
        
        # Normalize to [0.2, 1.0] range
        # (never completely zero to avoid harsh penalties)
        score = 0.2 + (action_freq * 0.8)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_resource_deviation(
        self,
        request: AccessRequest,
        baseline: Dict
    ) -> float:
        """
        Calculate resource usage deviation.
        
        Returns score in [0, 1] where 1 = normal, 0 = abnormal.
        """
        resource = request.resource
        resource_type = request.resource_type.value
        
        # Get normal resources from baseline
        normal_resources = baseline.get('normal_resources', {})
        resource_list = normal_resources.get('common_resources', [])
        resource_types = normal_resources.get('resource_type_distribution', {})
        
        # Check if resource is in normal list
        resource_in_list = resource in resource_list
        
        # Check if resource type is common
        resource_type_freq = resource_types.get(resource_type, 0.0)
        
        # Combined score
        if resource_in_list:
            score = 1.0  # Perfect match
        elif resource_type_freq > 0.1:  # Type is common (>10%)
            score = 0.7
        else:
            score = 0.3  # Unusual resource
        
        return float(score)
    
    def _calculate_temporal_deviation(
        self,
        request: AccessRequest,
        baseline: Dict,
        features: Optional[Dict]
    ) -> float:
        """
        Calculate temporal pattern deviation.
        
        Considers hour of day, day of week, time gaps.
        """
        if features is None:
            features = {}
        
        # Get temporal features
        hour_of_day = features.get('hour_of_day', 12)  # Default noon
        day_of_week = features.get('day_of_week', 3)  # Default Wednesday
        
        # Get baseline temporal patterns
        normal_usage = baseline.get('normal_usage', {})
        hourly_dist = normal_usage.get('hourly_distribution', {})
        daily_dist = normal_usage.get('daily_distribution', {})
        
        # Hour score
        hour_freq = hourly_dist.get(str(hour_of_day), 0.0)
        hour_score = 0.2 + (hour_freq * 0.8)
        
        # Day score
        day_freq = daily_dist.get(str(day_of_week), 0.0)
        day_score = 0.2 + (day_freq * 0.8)
        
        # Peak hours bonus (if defined)
        peak_hours = normal_usage.get('peak_hours', [])
        if hour_of_day in peak_hours:
            hour_score = min(1.0, hour_score * 1.2)
        
        # Combine (70% hour, 30% day)
        score = 0.7 * hour_score + 0.3 * day_score
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_location_deviation(
        self,
        request: AccessRequest,
        baseline: Dict
    ) -> float:
        """
        Calculate location usage deviation.
        """
        location = request.location
        
        # Get location distribution
        normal_usage = baseline.get('normal_usage', {})
        location_dist = normal_usage.get('location_diversity', {})
        
        if not location_dist:
            return 0.5  # Neutral
        
        # Check if location is in distribution
        location_freq = location_dist.get(location, 0.0)
        
        # Score based on frequency
        score = 0.2 + (location_freq * 0.8)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_frequency_deviation(
        self,
        request: AccessRequest,
        baseline: Dict,
        features: Optional[Dict]
    ) -> float:
        """
        Calculate request frequency deviation.
        
        Checks if request rate is normal.
        """
        if features is None:
            return 0.5
        
        # Get current request rate
        current_rate = features.get('request_rate', 0.0)
        
        # Get baseline rate
        normal_usage = baseline.get('normal_usage', {})
        avg_rate = normal_usage.get('avg_requests_per_day', 100) / (24 * 3600)  # Convert to per-second
        
        if avg_rate == 0:
            return 0.5
        
        # Calculate deviation (ratio)
        ratio = current_rate / avg_rate
        
        # Score based on how close to expected
        # Optimal ratio is 1.0, acceptable range is 0.5-2.0
        if 0.5 <= ratio <= 2.0:
            score = 1.0 - abs(ratio - 1.0) * 0.3  # Small penalty
        elif ratio < 0.5:
            score = 0.5 * (ratio / 0.5)  # Too slow
        else:  # ratio > 2.0
            score = 0.5 * (2.0 / ratio)  # Too fast
        
        return float(np.clip(score, 0.0, 1.0))
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update feature weights."""
        total = sum(new_weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        
        self.weights = new_weights
        logger.info(f"Updated weights: {new_weights}")
    
    def update_threshold(self, new_threshold: float):
        """Update decision threshold."""
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0,1], got {new_threshold}")
        
        self.threshold = new_threshold
        logger.info(f"Updated threshold: {new_threshold}")


class IsolationForestModel:
    """
    Isolation Forest for anomaly detection.
    
    Uses sklearn's IsolationForest on extracted features.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest model.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
        """
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all CPUs
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(
            f"IsolationForest initialized: contamination={contamination}, "
            f"n_estimators={n_estimators}"
        )
    
    def fit(self, features: np.ndarray):
        """
        Train Isolation Forest on feature vectors.
        
        Args:
            features: Feature matrix (n_samples, n_features)
        """
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(features_scaled)
        self.is_fitted = True
        
        logger.info(f"IsolationForest fitted on {len(features)} samples")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Anomaly scores (higher = more normal)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get anomaly scores
        scores = self.model.decision_function(features_scaled)
        
        # Normalize to [0, 1] (higher = more normal)
        # decision_function returns negative for anomalies
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return normalized
    
    def analyze(
        self,
        request: AccessRequest,
        features: Dict
    ) -> LayerDecision:
        """
        Analyze request using Isolation Forest.
        
        Args:
            request: Access request
            features: Extracted features
            
        Returns:
            LayerDecision with anomaly score
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted:
            return LayerDecision(
                decision=DecisionType.GRANT,
                confidence=0.5,
                latency_ms=(time.time() - start_time) * 1000,
                layer_name='isolation_forest',
                explanation={'reason': 'model_not_fitted'}
            )
        
        # Convert features to array
        feature_vector = self._features_to_vector(features)
        
        # Predict
        score = self.predict(feature_vector.reshape(1, -1))[0]
        
        # Decision (score >= 0.5 → normal)
        decision = (
            DecisionType.GRANT if score >= 0.5
            else DecisionType.DENY
        )
        
        confidence = float(score)
        latency_ms = (time.time() - start_time) * 1000
        
        return LayerDecision(
            decision=decision,
            confidence=confidence,
            latency_ms=float(latency_ms),
            layer_name='isolation_forest',
            explanation={
                'anomaly_score': float(score),
                'threshold': 0.5,
            }
        )
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array."""
        # Extract numerical features
        vector = [
            features.get('hour_of_day', 0),
            features.get('day_of_week', 0),
            features.get('total_accesses', 0),
            features.get('unique_resources', 0),
            features.get('emergency_rate', 0),
            features.get('human_present_rate', 0),
            features.get('avg_time_gap', 0),
            features.get('time_gap_std', 0),
            features.get('location_diversity', 0),
            features.get('action_repetition_rate', 0),
        ]
        
        return np.array(vector, dtype=np.float32)
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")


__all__ = ['StatisticalModel', 'IsolationForestModel']