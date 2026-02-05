#!/usr/bin/env python3
"""
BBAC ICS Framework - Fusion Models

Meta-classifier models for combining layer decisions:
- Logistic Regression
- XGBoost
- Feature extraction from layer outputs

Learns optimal fusion weights from historical data.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MetaClassifier:
    """
    Meta-classifier for learning optimal layer fusion.
    
    Takes layer decisions as features and predicts final decision.
    """
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        **model_params
    ):
        """
        Initialize meta-classifier.
        
        Args:
            model_type: 'logistic_regression' or 'xgboost'
            **model_params: Model-specific parameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=model_params.get('max_iter', 1000),
                random_state=model_params.get('random_state', 42),
                class_weight=model_params.get('class_weight', 'balanced'),
                C=model_params.get('C', 1.0),
            )
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    max_depth=model_params.get('max_depth', 3),
                    learning_rate=model_params.get('learning_rate', 0.1),
                    n_estimators=model_params.get('n_estimators', 100),
                    random_state=model_params.get('random_state', 42),
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                )
            except ImportError:
                logger.error("XGBoost not installed. Install with: pip install xgboost")
                raise
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        logger.info(f"MetaClassifier initialized: {model_type}")
    
    def extract_features(
        self,
        layer_results: Dict[str, Dict]
    ) -> np.ndarray:
        """
        Extract features from layer decisions.
        
        Args:
            layer_results: Dictionary of layer decisions
                {
                    'statistical': {'decision': 'grant', 'confidence': 0.8, ...},
                    'sequence': {'decision': 'deny', 'confidence': 0.6, ...},
                    'policy': {'decision': 'grant', 'confidence': 1.0, ...},
                }
        
        Returns:
            Feature vector
        """
        features = []
        
        # Layer names in fixed order
        layer_names = ['statistical', 'sequence', 'policy']
        
        for layer in layer_names:
            if layer in layer_results:
                result = layer_results[layer]
                
                # Decision as binary (grant=1, deny=0)
                decision = 1.0 if result.get('decision') == 'grant' else 0.0
                
                # Confidence
                confidence = result.get('confidence', 0.5)
                
                # Latency (normalized)
                latency = result.get('latency_ms', 50.0) / 100.0  # Normalize by 100ms
                
                features.extend([decision, confidence, latency])
            else:
                # Missing layer - use neutral values
                features.extend([0.5, 0.5, 0.5])
        
        return np.array(features, dtype=np.float32)
    
    def fit(
        self,
        layer_results_list: List[Dict[str, Dict]],
        ground_truth: List[str]
    ):
        """
        Train meta-classifier on historical data.
        
        Args:
            layer_results_list: List of layer decision dictionaries
            ground_truth: List of ground truth decisions ('grant' or 'deny')
        """
        # Extract features
        X = np.array([
            self.extract_features(lr)
            for lr in layer_results_list
        ])
        
        # Convert ground truth to binary
        y = np.array([1 if gt == 'grant' else 0 for gt in ground_truth])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Log accuracy
        train_score = self.model.score(X_scaled, y)
        logger.info(
            f"MetaClassifier fitted: {len(X)} samples, "
            f"train_accuracy={train_score:.4f}"
        )
        
        # Get feature importances (if available)
        if hasattr(self.model, 'coef_'):
            self._log_feature_importance(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            self._log_feature_importance(self.model.feature_importances_)
    
    def predict(
        self,
        layer_results: Dict[str, Dict]
    ) -> Tuple[str, float]:
        """
        Predict final decision using meta-classifier.
        
        Args:
            layer_results: Layer decision dictionary
            
        Returns:
            Tuple of (decision, confidence)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Extract features
        features = self.extract_features(layer_results).reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        
        # Convert to decision
        decision = 'grant' if prediction == 1 else 'deny'
        confidence = float(proba[prediction])
        
        return decision, confidence
    
    def _log_feature_importance(self, importances: np.ndarray):
        """Log feature importances."""
        feature_names = []
        layers = ['statistical', 'sequence', 'policy']
        for layer in layers:
            feature_names.extend([
                f'{layer}_decision',
                f'{layer}_confidence',
                f'{layer}_latency'
            ])
        
        # Sort by importance
        indices = np.argsort(np.abs(importances))[::-1]
        
        logger.info("Feature importances (top 5):")
        for i in range(min(5, len(indices))):
            idx = indices[i]
            logger.info(
                f"  {feature_names[idx]}: {importances[idx]:.4f}"
            )
    
    def get_learned_weights(self) -> Dict[str, float]:
        """
        Extract layer weights from trained model.
        
        Returns:
            Dictionary with layer weights
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_[0]
        elif hasattr(self.model, 'feature_importances_'):
            coefs = self.model.feature_importances_
        else:
            logger.warning("Cannot extract weights from this model")
            return {'statistical': 0.33, 'sequence': 0.33, 'policy': 0.34}
        
        # Average coefficients for each layer (3 features per layer)
        weights = {
            'statistical': float(np.mean(np.abs(coefs[0:3]))),
            'sequence': float(np.mean(np.abs(coefs[3:6]))),
            'policy': float(np.mean(np.abs(coefs[6:9]))),
        }
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        logger.info(f"Learned weights: {weights}")
        
        return weights
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'model_type': self.model_type,
            }, f)
        
        logger.info(f"MetaClassifier saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        self.model_type = data['model_type']
        
        logger.info(f"MetaClassifier loaded from {filepath}")


class EnsembleFusion:
    """
    Ensemble of multiple meta-classifiers.
    
    Combines predictions from multiple models for robustness.
    """
    
    def __init__(
        self,
        models: Optional[List[MetaClassifier]] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of MetaClassifier instances
        """
        if models is None:
            # Create default ensemble: LogReg + XGBoost
            try:
                models = [
                    MetaClassifier('logistic_regression'),
                    MetaClassifier('xgboost'),
                ]
            except ImportError:
                # XGBoost not available, use only LogReg
                logger.warning("XGBoost not available, using only LogisticRegression")
                models = [MetaClassifier('logistic_regression')]
        
        self.models = models
        logger.info(f"EnsembleFusion initialized with {len(models)} models")
    
    def fit(
        self,
        layer_results_list: List[Dict[str, Dict]],
        ground_truth: List[str]
    ):
        """Train all models in ensemble."""
        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{len(self.models)}")
            model.fit(layer_results_list, ground_truth)
    
    def predict(
        self,
        layer_results: Dict[str, Dict],
        voting: str = 'soft'
    ) -> Tuple[str, float]:
        """
        Predict using ensemble voting.
        
        Args:
            layer_results: Layer decision dictionary
            voting: 'hard' (majority vote) or 'soft' (average probabilities)
            
        Returns:
            Tuple of (decision, confidence)
        """
        predictions = []
        confidences = []
        
        for model in self.models:
            decision, confidence = model.predict(layer_results)
            predictions.append(decision)
            confidences.append(confidence)
        
        if voting == 'hard':
            # Majority vote
            grant_count = sum(1 for d in predictions if d == 'grant')
            final_decision = 'grant' if grant_count > len(predictions) / 2 else 'deny'
            
            # Average confidence of models that agree
            agreeing_confidences = [
                c for d, c in zip(predictions, confidences)
                if d == final_decision
            ]
            final_confidence = float(np.mean(agreeing_confidences))
            
        else:  # soft voting
            # Average confidence scores
            grant_scores = []
            deny_scores = []
            
            for decision, confidence in zip(predictions, confidences):
                if decision == 'grant':
                    grant_scores.append(confidence)
                    deny_scores.append(1.0 - confidence)
                else:
                    grant_scores.append(1.0 - confidence)
                    deny_scores.append(confidence)
            
            avg_grant = float(np.mean(grant_scores))
            avg_deny = float(np.mean(deny_scores))
            
            if avg_grant > avg_deny:
                final_decision = 'grant'
                final_confidence = avg_grant
            else:
                final_decision = 'deny'
                final_confidence = avg_deny
        
        return final_decision, final_confidence
    
    def get_averaged_weights(self) -> Dict[str, float]:
        """Get average weights across all models."""
        all_weights = [model.get_learned_weights() for model in self.models]
        
        # Average weights
        avg_weights = {
            'statistical': float(np.mean([w['statistical'] for w in all_weights])),
            'sequence': float(np.mean([w['sequence'] for w in all_weights])),
            'policy': float(np.mean([w['policy'] for w in all_weights])),
        }
        
        # Normalize
        total = sum(avg_weights.values())
        avg_weights = {k: v/total for k, v in avg_weights.items()}
        
        return avg_weights


__all__ = ['MetaClassifier', 'EnsembleFusion']