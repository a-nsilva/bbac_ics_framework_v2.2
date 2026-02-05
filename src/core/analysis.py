#!/usr/bin/env python3
"""
BBAC ICS Framework - Analysis Layer

Implements:
- Statistical Engine (deviation analysis)
- Sequence Engine (Markov chain analysis)
- Policy Engine (RuBAC rules)

From flowchart:
- Branch 1: Statistical analysis (frequency, deviation)
- Branch 2: Sequence model (LSTM/Markov)
- Policy Engine: RuBAC compliance

Layer 3: Analysis (Statistical + Sequence + Policy)
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models.statistical import StatisticalModel, IsolationForestModel
from ..models.lstm import LSTMSequenceModel
from ..util.config_loader import config
from ..util.data_structures import (
    AccessRequest,
    LayerDecision,
    DecisionType,
    ActionType,
)


logger = logging.getLogger(__name__)


class StatisticalEngine:
    """
    Statistical analysis engine.
    
    Uses StatisticalModel from models/statistical.py
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        use_isolation_forest: bool = False,
        isolation_forest_path: Optional[str] = None
    ):
        """
        Initialize statistical engine.
        
        Args:
            threshold: Decision threshold
            use_isolation_forest: Use Isolation Forest for anomaly detection
            isolation_forest_path: Path to trained Isolation Forest model
        """
        # Primary model: Statistical
        self.statistical_model = StatisticalModel(threshold=threshold)
        self.threshold = threshold
        
        # Optional: Isolation Forest
        self.use_isolation_forest = use_isolation_forest
        self.isolation_forest = None
        
        if use_isolation_forest and isolation_forest_path:
            try:
                self.isolation_forest = IsolationForestModel()
                self.isolation_forest.load_model(isolation_forest_path)
                logger.info(f"Loaded Isolation Forest from {isolation_forest_path}")
            except Exception as e:
                logger.error(f"Failed to load Isolation Forest: {e}")
                self.use_isolation_forest = False
        
        logger.info(
            f"StatisticalEngine initialized: threshold={threshold}, "
            f"isolation_forest={use_isolation_forest}"
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
            features: Pre-extracted features
            
        Returns:
            LayerDecision
        """
        # Use statistical model
        statistical_result = self.statistical_model.analyze(
            request, baseline, features
        )
        
        # If Isolation Forest is enabled, combine results
        if self.use_isolation_forest and self.isolation_forest and features:
            try:
                isolation_result = self.isolation_forest.analyze(request, features)
                
                # Combine scores (70% statistical, 30% isolation forest)
                combined_score = (
                    0.7 * (1.0 if statistical_result.decision == DecisionType.GRANT else 0.0) +
                    0.3 * isolation_result.confidence
                )
                
                combined_decision = (
                    DecisionType.GRANT if combined_score >= 0.5
                    else DecisionType.DENY
                )
                
                combined_confidence = abs(combined_score - 0.5) * 2
                
                return LayerDecision(
                    decision=combined_decision,
                    confidence=float(combined_confidence),
                    latency_ms=statistical_result.latency_ms + isolation_result.latency_ms,
                    layer_name='statistical_combined',
                    explanation={
                        'statistical': statistical_result.explanation,
                        'isolation_forest': isolation_result.explanation,
                        'combined_score': float(combined_score),
                    }
                )
                
            except Exception as e:
                logger.error(f"Error in Isolation Forest: {e}")
                # Fallback to statistical only
                return statistical_result
        
        return statistical_result
    
    def update_threshold(self, new_threshold: float):
        """Update decision threshold."""
        self.threshold = new_threshold
        self.statistical_model.update_threshold(new_threshold)


class SequenceEngine:
    """
    Sequence analysis engine.
    
    Supports both Markov chains and LSTM models.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        use_lstm: bool = False,
        lstm_path: Optional[str] = None
    ):
        """
        Initialize sequence engine.
        
        Args:
            threshold: Decision threshold
            use_lstm: Use LSTM model instead of Markov chains
            lstm_path: Path to trained LSTM model
        """
        self.threshold = threshold
        self.use_lstm = use_lstm
        
        # Markov chains (default)
        self.transition_matrices = {}  # agent_id -> transition matrix
        
        # LSTM (optional)
        self.lstm_model = None
        
        if use_lstm and lstm_path:
            try:
                self.lstm_model = LSTMSequenceModel()
                self.lstm_model.load_model(lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")
                self.use_lstm = False
        
        logger.info(
            f"SequenceEngine initialized: threshold={threshold}, "
            f"use_lstm={use_lstm}"
        )
    
    def build_transition_matrix(
        self,
        agent_id: str,
        data
    ):
        """
        Build Markov transition matrix for agent.
        
        Args:
            agent_id: Agent identifier
            data: Historical data (DataFrame or list)
        """
        import pandas as pd
        from collections import defaultdict
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))
        
        actions = data['action'].tolist()
        
        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]
            transitions[current][next_action] += 1
        
        # Normalize to probabilities
        transition_matrix = {}
        
        for current, nexts in transitions.items():
            total = sum(nexts.values())
            transition_matrix[current] = {
                next_action: count / total
                for next_action, count in nexts.items()
            }
        
        self.transition_matrices[agent_id] = transition_matrix
        
        logger.info(f"Built transition matrix for {agent_id}")
    
    def analyze(
        self,
        request: AccessRequest,
        baseline: Optional[Dict] = None,
        features: Optional[Dict] = None
    ) -> LayerDecision:
        """
        Analyze request using sequence analysis.
        
        Args:
            request: Access request
            baseline: Agent baseline (unused for now)
            features: Pre-extracted features
            
        Returns:
            LayerDecision
        """
        start_time = time.time()
        
        # Get previous action
        if request.previous_action is None:
            # No previous action - neutral decision
            return LayerDecision(
                decision=DecisionType.GRANT,
                confidence=0.5,
                latency_ms=(time.time() - start_time) * 1000,
                layer_name='sequence',
                explanation={'reason': 'no_previous_action'}
            )
        
        # Use LSTM if available
        if self.use_lstm and self.lstm_model:
            return self._analyze_with_lstm(request, start_time)
        
        # Otherwise use Markov chains
        return self._analyze_with_markov(request, start_time)
    
    def _analyze_with_markov(
        self,
        request: AccessRequest,
        start_time: float
    ) -> LayerDecision:
        """Analyze using Markov chains."""
        agent_id = request.agent_id
        previous_action = request.previous_action.value
        current_action = request.action.value
        
        # Get transition matrix
        if agent_id not in self.transition_matrices:
            # No model for agent - neutral
            return LayerDecision(
                decision=DecisionType.GRANT,
                confidence=0.5,
                latency_ms=(time.time() - start_time) * 1000,
                layer_name='sequence_markov',
                explanation={'reason': 'no_transition_matrix'}
            )
        
        matrix = self.transition_matrices[agent_id]
        
        # Get transition probability
        if previous_action not in matrix:
            probability = 0.0
        else:
            probability = matrix[previous_action].get(current_action, 0.0)
        
        # Map probability to score and decision
        # High probability → high score → grant
        if probability > 0.5:
            score = 0.9
        elif probability > 0.2:
            score = 0.7
        elif probability > 0.05:
            score = 0.5
        elif probability > 0:
            score = 0.3
        else:
            score = 0.1
        
        decision = DecisionType.GRANT if score >= self.threshold else DecisionType.DENY
        confidence = abs(score - 0.5) * 2  # Distance from 0.5
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LayerDecision(
            decision=decision,
            confidence=float(confidence),
            latency_ms=float(latency_ms),
            layer_name='sequence_markov',
            explanation={
                'transition_probability': float(probability),
                'score': float(score),
                'previous_action': previous_action,
                'current_action': current_action,
            }
        )
    
    def _analyze_with_lstm(
        self,
        request: AccessRequest,
        start_time: float
    ) -> LayerDecision:
        """Analyze using LSTM model."""
        # Build action sequence
        # In real implementation, this would come from session history
        # For now, use previous_action as minimal sequence
        action_sequence = [request.previous_action.value] if request.previous_action else []
        current_action = request.action.value
        
        # Analyze with LSTM
        result = self.lstm_model.analyze_sequence(
            action_sequence,
            current_action,
            threshold=self.threshold
        )
        
        # Convert to LayerDecision
        decision = DecisionType.GRANT if result['decision'] == 'grant' else DecisionType.DENY
        
        return LayerDecision(
            decision=decision,
            confidence=result['confidence'],
            latency_ms=result['latency_ms'],
            layer_name='sequence_lstm',
            explanation={
                'normalcy_score': result['normalcy_score'],
                'expected_actions': result.get('expected_actions', []),
                'actual_action': result['actual_action'],
            }
        )


class PolicyEngine:
    """
    Rule-based policy engine (RuBAC).
    
    No changes - already complete.
    """
    
    def __init__(self):
        """Initialize policy engine with default rules."""
        self.rules = []
        self.emergency_mode = False
        
        # Add default rules
        self._add_default_rules()
        
        logger.info(f"PolicyEngine initialized with {len(self.rules)} rules")
    
    def _add_default_rules(self):
        """Add default policy rules."""
        # Rule 1: Emergency override (highest priority)
        self.add_rule(
            rule_id='emergency_override',
            priority=1,
            condition=lambda req: req.emergency,
            action=DecisionType.GRANT,
            description='Grant access during emergencies'
        )
        
        # Rule 2: Critical actions require human supervision
        self.add_rule(
            rule_id='human_present_required',
            priority=2,
            condition=lambda req: (
                req.agent_type.value == 'robot' and
                req.action in [ActionType.DELETE, ActionType.OVERRIDE] and
                not req.human_present
            ),
            action=DecisionType.DENY,
            description='Critical robot actions require human supervision'
        )
        
        # Rule 3: Authentication failures
        self.add_rule(
            rule_id='auth_failure_deny',
            priority=3,
            condition=lambda req: req.attempt_count > 3,
            action=DecisionType.DENY,
            description='Deny after multiple auth failures'
        )
        
        # Rule 4: Supervisor override
        self.add_rule(
            rule_id='supervisor_override',
            priority=4,
            condition=lambda req: 'supervisor' in req.agent_role.value.lower(),
            action=DecisionType.GRANT,
            description='Supervisors have elevated privileges'
        )
        
        # Rule 5: Maintenance hours
        maintenance_hours = config.policy.get('maintenance_hours', [2, 3])
        
        self.add_rule(
            rule_id='maintenance_hours',
            priority=5,
            condition=lambda req: self._is_maintenance_hours(req),
            action=DecisionType.REQUIRE_APPROVAL,
            description='Require approval during maintenance hours'
        )
    
    def _is_maintenance_hours(self, request: AccessRequest) -> bool:
        """Check if request is during maintenance hours."""
        import datetime
        
        maintenance_hours = config.policy.get('maintenance_hours', [2, 3])
        
        # Get hour from timestamp
        if hasattr(request, 'timestamp'):
            dt = datetime.datetime.fromtimestamp(request.timestamp)
            hour = dt.hour
            return hour in maintenance_hours
        
        return False
    
    def add_rule(
        self,
        rule_id: str,
        priority: int,
        condition: callable,
        action: DecisionType,
        description: str = ''
    ):
        """Add new rule."""
        rule = {
            'id': rule_id,
            'priority': priority,
            'condition': condition,
            'action': action,
            'description': description,
        }
        
        self.rules.append(rule)
        
        # Sort by priority
        self.rules.sort(key=lambda r: r['priority'])
        
        logger.info(f"Added rule: {rule_id} (priority={priority})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove rule by ID."""
        initial_len = len(self.rules)
        self.rules = [r for r in self.rules if r['id'] != rule_id]
        
        removed = len(self.rules) < initial_len
        
        if removed:
            logger.info(f"Removed rule: {rule_id}")
        
        return removed
    
    def analyze(
        self,
        request: AccessRequest,
        baseline: Optional[Dict] = None,
        features: Optional[Dict] = None
    ) -> LayerDecision:
        """
        Analyze request using policy rules.
        
        Args:
            request: Access request
            baseline: Unused
            features: Unused
            
        Returns:
            LayerDecision
        """
        import time
        start_time = time.time()
        
        # Check rules in priority order
        for rule in self.rules:
            try:
                if rule['condition'](request):
                    # Rule matched
                    latency_ms = (time.time() - start_time) * 1000
                    
                    return LayerDecision(
                        decision=rule['action'],
                        confidence=1.0,  # Rules are deterministic
                        latency_ms=float(latency_ms),
                        layer_name='policy',
                        explanation={
                            'matched_rule': rule['id'],
                            'description': rule['description'],
                            'priority': rule['priority'],
                        }
                    )
            except Exception as e:
                logger.error(f"Error evaluating rule {rule['id']}: {e}")
                continue
        
        # No rules matched - neutral (allow)
        latency_ms = (time.time() - start_time) * 1000
        
        return LayerDecision(
            decision=DecisionType.GRANT,
            confidence=0.5,
            latency_ms=float(latency_ms),
            layer_name='policy',
            explanation={'reason': 'no_rules_matched'}
        )
    
    def trigger_emergency(self, emergency_type: str) -> bool:
        """Trigger emergency mode."""
        self.emergency_mode = True
        logger.warning(f"Emergency mode triggered: {emergency_type}")
        return True
    
    def clear_emergency(self) -> bool:
        """Clear emergency mode."""
        self.emergency_mode = False
        logger.info("Emergency mode cleared")
        return True


class AnalysisLayer:
    """
    Analysis Layer - orchestrates all analysis engines.
    
    UPDATED: Uses new models from models/ directory.
    """
    
    def __init__(
        self,
        use_lstm: bool = False,
        use_isolation_forest: bool = False,
        models_dir: str = 'trained_models'
    ):
        """
        Initialize analysis layer.
        
        Args:
            use_lstm: Use LSTM for sequence analysis
            use_isolation_forest: Use Isolation Forest for anomaly detection
            models_dir: Directory with trained models
        """
        from pathlib import Path
        
        models_path = Path(models_dir)
        
        # Initialize engines
        self.statistical_engine = StatisticalEngine(
            threshold=config.ml_params.get('statistical', {}).get('anomaly_threshold', 0.5),
            use_isolation_forest=use_isolation_forest,
            isolation_forest_path=str(models_path / 'isolation_forest.pkl') if use_isolation_forest else None
        )
        
        self.sequence_engine = SequenceEngine(
            threshold=config.ml_params.get('sequence', {}).get('anomaly_threshold', 0.5),
            use_lstm=use_lstm,
            lstm_path=str(models_path / 'lstm_sequence_model') if use_lstm else None
        )
        
        self.policy_engine = PolicyEngine()
        
        logger.info(
            f"AnalysisLayer initialized: lstm={use_lstm}, "
            f"isolation_forest={use_isolation_forest}"
        )
    
    def analyze_request(
        self,
        request: AccessRequest,
        profile_baseline: Optional[Dict],
        features: Optional[Dict],
        enable_stat: bool = True,
        enable_sequence: bool = True,
        enable_policy: bool = True,
    ) -> Dict[str, LayerDecision]:
        """
        Analyze request through all enabled engines.
        
        Args:
            request: Access request
            profile_baseline: Agent baseline
            features: Extracted features
            enable_stat: Enable statistical analysis
            enable_sequence: Enable sequence analysis
            enable_policy: Enable policy analysis
            
        Returns:
            Dictionary mapping engine names to LayerDecision
        """
        results = {}
        
        if enable_stat:
            results['statistical'] = self.statistical_engine.analyze(
                request, profile_baseline, features
            )
        
        if enable_sequence:
            results['sequence'] = self.sequence_engine.analyze(
                request, profile_baseline, features
            )
        
        if enable_policy:
            results['policy'] = self.policy_engine.analyze(
                request, profile_baseline, features
            )
        
        return results


__all__ = ['AnalysisLayer', 'StatisticalEngine', 'SequenceEngine', 'PolicyEngine']