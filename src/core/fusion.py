#!/usr/bin/env python3
"""
BBAC ICS Framework - Fusion Layer

Implements:
- Score Fusion Layer (combines stat, sequence, policy scores)
- Multiple fusion strategies
- Future: Meta-classifier support

From flowchart:
- Receives: stat_score, ml_score (sequence), policy_score
- Outputs: Fused score for decision layer

Layer 4a: Score Fusion
Combines decisions from multiple analysis layers.

Updated to use MetaClassifier from models/fusion.py
"""

import logging
import time
from typing import Dict, Optional

from ..util.config_loader import config
from ..util.data_structures import (
    DecisionType,
    FusionStrategy,
    LayerDecision,
    LayerWeights,
)
from ..models.fusion import MetaClassifier, EnsembleFusion


logger = logging.getLogger(__name__)


class FusionEngine:
    """
    Fusion engine with multiple strategies.
    
    Strategies:
    1. Weighted voting (default)
    2. Rule priority (policy veto)
    3. High confidence denial (any layer can block)
    """
    
    def __init__(
        self,
        strategy: str = 'weighted_voting',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize fusion engine.
        
        Args:
            strategy: Fusion strategy name
            weights: Layer weights for weighted voting
        """
        # Parse strategy
        if isinstance(strategy, str):
            try:
                self.strategy = FusionStrategy.from_string(strategy)
            except ValueError:
                logger.warning(f"Unknown strategy '{strategy}', using weighted_voting")
                self.strategy = FusionStrategy.WEIGHTED_VOTING
        else:
            self.strategy = strategy
        
        # Default weights from config
        if weights is None:
            fusion_config = config.fusion
            weights = fusion_config.get('weights', {
                'rule': 0.4,
                'behavioral': 0.3,
                'ml': 0.3,
            })
        
        self.weights = LayerWeights(
            rule=weights.get('rule', 0.4),
            behavioral=weights.get('behavioral', 0.3),
            ml=weights.get('ml', 0.3)
        )
        
        # Thresholds
        self.decision_threshold = config.fusion.get('decision_threshold', 0.5)
        self.high_confidence_threshold = config.fusion.get('high_confidence_threshold', 0.9)
        
        logger.info(
            f"FusionEngine initialized: strategy={self.strategy.value}, "
            f"weights={self.weights}"
        )
    
    def fuse(
        self,
        layer_results: Dict[str, LayerDecision]
    ) -> Dict:
        """
        Fuse layer decisions using configured strategy.
        
        Args:
            layer_results: Dictionary of LayerDecision objects
                {
                    'statistical': LayerDecision(...),
                    'sequence': LayerDecision(...),
                    'policy': LayerDecision(...),
                }
        
        Returns:
            Fusion result dictionary
        """
        start_time = time.time()
        
        if not layer_results:
            # No layers - neutral decision
            return {
                'decision': 'grant',
                'confidence': 0.5,
                'total_latency_ms': 0.0,
                'strategy': self.strategy.value,
                'layer_results': {},
            }
        
        # Apply fusion strategy
        if self.strategy == FusionStrategy.WEIGHTED_VOTING:
            result = self._weighted_voting(layer_results)
        elif self.strategy == FusionStrategy.RULE_PRIORITY:
            result = self._rule_priority(layer_results)
        elif self.strategy == FusionStrategy.HIGH_CONFIDENCE_DENIAL:
            result = self._high_confidence_denial(layer_results)
        else:
            logger.error(f"Unknown strategy: {self.strategy}")
            result = self._weighted_voting(layer_results)
        
        # Add total latency
        total_latency = sum(d.latency_ms for d in layer_results.values())
        result['total_latency_ms'] = total_latency + (time.time() - start_time) * 1000
        
        # Add strategy
        result['strategy'] = self.strategy.value
        
        # Convert LayerDecision to dict for serialization
        result['layer_results'] = {
            name: {
                'decision': decision.decision.value,
                'confidence': decision.confidence,
                'latency_ms': decision.latency_ms,
                'explanation': decision.explanation,
            }
            for name, decision in layer_results.items()
        }
        
        return result
    
    def _weighted_voting(
        self,
        layer_results: Dict[str, LayerDecision]
    ) -> Dict:
        """
        Weighted voting fusion strategy.
        
        Formula: score = Σ(weight_i * decision_i * confidence_i)
        Decision: grant if score >= threshold
        """
        # Map layer names to weight categories
        layer_weight_map = {
            'policy': 'rule',
            'statistical': 'behavioral',
            'sequence': 'ml',
            'sequence_markov': 'ml',
            'sequence_lstm': 'ml',
            'statistical_combined': 'behavioral',
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for layer_name, decision in layer_results.items():
            # Get weight
            weight_category = layer_weight_map.get(layer_name, 'behavioral')
            weight = getattr(self.weights, weight_category, 0.3)
            
            # Convert decision to score (grant=1.0, deny=0.0)
            decision_score = 1.0 if decision.decision == DecisionType.GRANT else 0.0
            
            # Weight by confidence
            weighted_score = weight * decision_score * decision.confidence
            
            total_score += weighted_score
            total_weight += weight * decision.confidence
        
        # Normalize score
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            normalized_score = 0.5  # Neutral
        
        # Make decision
        final_decision = 'grant' if normalized_score >= self.decision_threshold else 'deny'
        
        # Calculate confidence (distance from threshold)
        if final_decision == 'grant':
            confidence = 0.5 + (normalized_score - self.decision_threshold) / (1.0 - self.decision_threshold) * 0.5
        else:
            confidence = 0.5 + (self.decision_threshold - normalized_score) / self.decision_threshold * 0.5
        
        return {
            'decision': final_decision,
            'confidence': float(confidence),
            'normalized_score': float(normalized_score),
            'threshold': self.decision_threshold,
        }
    
    def _rule_priority(
        self,
        layer_results: Dict[str, LayerDecision]
    ) -> Dict:
        """
        Rule priority fusion strategy.
        
        Policy layer has veto power:
        - If policy denies → final deny
        - Otherwise → weighted voting
        """
        # Check policy layer
        if 'policy' in layer_results:
            policy_decision = layer_results['policy']
            
            if policy_decision.decision == DecisionType.DENY:
                # Policy veto
                return {
                    'decision': 'deny',
                    'confidence': policy_decision.confidence,
                    'reason': 'policy_veto',
                    'policy_explanation': policy_decision.explanation,
                }
        
        # Otherwise use weighted voting
        return self._weighted_voting(layer_results)
    
    def _high_confidence_denial(
        self,
        layer_results: Dict[str, LayerDecision]
    ) -> Dict:
        """
        High confidence denial fusion strategy.
        
        Any layer with deny + high confidence can block:
        - If any layer: deny AND confidence >= threshold → final deny
        - Otherwise → weighted voting
        """
        for layer_name, decision in layer_results.items():
            if (decision.decision == DecisionType.DENY and 
                decision.confidence >= self.high_confidence_threshold):
                # High confidence denial
                return {
                    'decision': 'deny',
                    'confidence': decision.confidence,
                    'reason': 'high_confidence_denial',
                    'blocking_layer': layer_name,
                    'layer_explanation': decision.explanation,
                }
        
        # No high confidence denials, use weighted voting
        return self._weighted_voting(layer_results)
    
    def update_weights(
        self,
        rule: Optional[float] = None,
        behavioral: Optional[float] = None,
        ml: Optional[float] = None
    ):
        """Update layer weights."""
        if rule is not None:
            self.weights.rule = rule
        if behavioral is not None:
            self.weights.behavioral = behavioral
        if ml is not None:
            self.weights.ml = ml
        
        # Validate sum to 1.0
        total = self.weights.rule + self.weights.behavioral + self.weights.ml
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            logger.warning(f"Weights sum to {total:.3f}, not 1.0. Normalizing...")
            self.weights.rule /= total
            self.weights.behavioral /= total
            self.weights.ml /= total
        
        logger.info(f"Updated weights: {self.weights}")
    
    def update_strategy(self, strategy: str):
        """Update fusion strategy."""
        try:
            self.strategy = FusionStrategy.from_string(strategy)
            logger.info(f"Updated strategy: {strategy}")
        except ValueError:
            logger.error(f"Unknown strategy: {strategy}")


class MetaClassifierFusion:
    """
    Meta-classifier based fusion strategy.
    
    Uses trained model to learn optimal fusion from data.
    """
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        use_ensemble: bool = False,
        model_path: Optional[str] = None
    ):
        """
        Initialize meta-classifier fusion.
        
        Args:
            model_type: 'logistic_regression' or 'xgboost'
            use_ensemble: Use ensemble of models
            model_path: Path to trained model
        """
        self.use_ensemble = use_ensemble
        self.model_type = model_type
        
        # Initialize classifier
        if use_ensemble:
            self.classifier = EnsembleFusion()
        else:
            self.classifier = MetaClassifier(model_type)
        
        # Load trained model if path provided
        if model_path:
            try:
                self.classifier.load_model(model_path)
                logger.info(f"Loaded meta-classifier from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load meta-classifier: {e}")
        
        logger.info(
            f"MetaClassifierFusion initialized: "
            f"type={model_type}, ensemble={use_ensemble}"
        )
    
    def fuse(
        self,
        layer_results: Dict[str, LayerDecision]
    ) -> Dict:
        """
        Fuse layer results using meta-classifier.
        
        Args:
            layer_results: Dictionary of LayerDecision objects
            
        Returns:
            Fusion result dictionary
        """
        import time
        start_time = time.time()
        
        # Convert LayerDecision to dict format for classifier
        results_dict = {}
        for layer_name, layer_decision in layer_results.items():
            results_dict[layer_name] = {
                'decision': layer_decision.decision.value,
                'confidence': layer_decision.confidence,
                'latency_ms': layer_decision.latency_ms,
            }
        
        # Predict
        try:
            if self.use_ensemble:
                decision, confidence = self.classifier.predict(
                    results_dict,
                    voting='soft'
                )
            else:
                decision, confidence = self.classifier.predict(results_dict)
            
        except RuntimeError:
            # Model not fitted - fallback to weighted voting
            logger.warning("MetaClassifier not fitted, using weighted voting fallback")
            return self._fallback_fusion(layer_results)
        
        total_latency = sum(d.latency_ms for d in layer_results.values())
        total_latency += (time.time() - start_time) * 1000
        
        return {
            'decision': decision,
            'confidence': confidence,
            'total_latency_ms': total_latency,
            'strategy': 'meta_classifier',
            'layer_results': results_dict,
        }
    
    def _fallback_fusion(self, layer_results: Dict[str, LayerDecision]) -> Dict:
        """Fallback to weighted voting if model not fitted."""
        # Use default weights
        weights = {'statistical': 0.3, 'sequence': 0.3, 'policy': 0.4}
        
        score = 0.0
        total_weight = 0.0
        
        for layer_name, decision in layer_results.items():
            weight = weights.get(layer_name, 0.0)
            layer_score = 1.0 if decision.decision == DecisionType.GRANT else 0.0
            layer_score *= decision.confidence
            
            score += weight * layer_score
            total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        return {
            'decision': 'grant' if score >= 0.5 else 'deny',
            'confidence': abs(score - 0.5) * 2,
            'total_latency_ms': sum(d.latency_ms for d in layer_results.values()),
            'strategy': 'weighted_voting_fallback',
            'layer_results': {
                name: {
                    'decision': d.decision.value,
                    'confidence': d.confidence,
                    'latency_ms': d.latency_ms,
                }
                for name, d in layer_results.items()
            },
        }
    
    def get_learned_weights(self) -> Dict[str, float]:
        """Get learned layer weights from model."""
        if self.use_ensemble:
            return self.classifier.get_averaged_weights()
        else:
            return self.classifier.get_learned_weights()


class FusionLayer:
    """
    Fusion Layer - orchestrates fusion strategies.
    
    UPDATED: Supports both traditional and meta-classifier fusion.
    """
    
    def __init__(
        self,
        use_meta_classifier: bool = False,
        meta_model_type: str = 'logistic_regression',
        use_ensemble: bool = False,
        models_dir: str = 'trained_models'
    ):
        """
        Initialize fusion layer.
        
        Args:
            use_meta_classifier: Use meta-classifier instead of traditional fusion
            meta_model_type: 'logistic_regression' or 'xgboost'
            use_ensemble: Use ensemble meta-classifier
            models_dir: Directory with trained models
        """
        from pathlib import Path
        
        self.use_meta_classifier = use_meta_classifier
        models_path = Path(models_dir)
        
        if use_meta_classifier:
            # Use meta-classifier fusion
            if use_ensemble:
                model_path = None  # Ensemble loads both models internally
            else:
                model_file = 'meta_logreg.pkl' if meta_model_type == 'logistic_regression' else 'meta_xgboost.pkl'
                model_path = str(models_path / model_file)
            
            self.fusion = MetaClassifierFusion(
                model_type=meta_model_type,
                use_ensemble=use_ensemble,
                model_path=model_path
            )
        else:
            # Use traditional fusion
            strategy = config.fusion.get('fusion_method', 'weighted_voting')
            self.fusion = FusionEngine(strategy=strategy)
        
        logger.info(
            f"FusionLayer initialized: meta_classifier={use_meta_classifier}"
        )
    
    def fuse(
        self,
        layer_results: Dict[str, LayerDecision]
    ) -> Dict:
        """
        Fuse layer results.
        
        Args:
            layer_results: Dictionary of LayerDecision objects
            
        Returns:
            Fusion result dictionary
        """
        return self.fusion.fuse(layer_results)
    
    def update_weights(self, **kwargs):
        """Update fusion weights (only for traditional fusion)."""
        if not self.use_meta_classifier and isinstance(self.fusion, FusionEngine):
            self.fusion.update_weights(**kwargs)
        else:
            logger.warning("Cannot update weights for meta-classifier fusion")
    
    def update_strategy(self, strategy: str):
        """Update fusion strategy (only for traditional fusion)."""
        if not self.use_meta_classifier and isinstance(self.fusion, FusionEngine):
            self.fusion.update_strategy(strategy)
        else:
            logger.warning("Cannot update strategy for meta-classifier fusion")


__all__ = ['FusionLayer', 'FusionEngine', 'MetaClassifierFusion']