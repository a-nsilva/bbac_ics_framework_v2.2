#!/usr/bin/env python3
"""
BBAC ICS Framework - Decision Layer

Implements:
- Decision Engine (risk-based thresholds)
- RBAC Enforcement (final access control)
- Decision logging and auditing

From flowchart:
- Inputs: Fused score from fusion layer
- Outputs: 
  - low < T1 → Allow
  - T1 ≤ T2 → MFA (Multi-Factor Authentication)
  - T2 < T3 → Manual Review
  - ≥ T3 → Deny
"""

import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..util.config_loader import config
from ..util.data_structures import (
    AccessRequest,
    AccessDecision,
    DecisionType,
    LayerDecision,
)


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for decision thresholds."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionEngine:
    """
    Makes final access control decisions based on fused scores.
    
    Implements graduated response based on risk thresholds.
    """
    
    def __init__(self):
        """Initialize decision engine with thresholds."""
        self.thresholds_config = config.thresholds
        
        # Load thresholds (scores represent "normalcy" - high score = low risk)
        # Thresholds define boundaries between risk levels
        self.t1 = self.thresholds_config.get('t1_allow', 0.7)      # Above: Allow
        self.t2 = self.thresholds_config.get('t2_mfa', 0.5)        # Above: MFA
        self.t3 = self.thresholds_config.get('t3_review', 0.3)     # Above: Manual Review
        # Below T3: Deny
        
        logger.info(
            f"DecisionEngine initialized: T1={self.t1} (allow), "
            f"T2={self.t2} (mfa), T3={self.t3} (review)"
        )
    
    def decide(
        self,
        fused_score: float,
        fused_confidence: float,
        request: AccessRequest,
        fusion_explanation: Dict
    ) -> AccessDecision:
        """
        Make final decision based on fused score.
        
        Args:
            fused_score: Fused score from fusion layer [0.0, 1.0]
            fused_confidence: Confidence in fused score
            request: Original access request
            fusion_explanation: Explanation from fusion layer
            
        Returns:
            AccessDecision with final verdict
        """
        start_time = time.time()
        
        # Determine risk level and decision
        risk_level, decision_type = self._classify_risk(fused_score)
        
        # Generate reason
        reason = self._generate_reason(
            risk_level, 
            decision_type, 
            fused_score,
            fusion_explanation
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Add fusion latency if available
        total_latency_ms = latency_ms
        if 'fusion_latency_ms' in fusion_explanation:
            total_latency_ms += fusion_explanation['fusion_latency_ms']
        
        # Create decision
        access_decision = AccessDecision(
            request_id=request.request_id,
            timestamp=time.time(),
            decision=decision_type,
            confidence=fused_confidence,
            latency_ms=total_latency_ms,
            reason=reason,
            layer_decisions={
                'fusion': fusion_explanation,
                'decision': {
                    'fused_score': fused_score,
                    'risk_level': risk_level.value,
                    'thresholds': {
                        't1_allow': self.t1,
                        't2_mfa': self.t2,
                        't3_review': self.t3,
                    }
                }
            }
        )
        
        logger.info(
            f"Decision for {request.request_id}: {decision_type.value} "
            f"(score={fused_score:.3f}, risk={risk_level.value})"
        )
        
        return access_decision
    
    def _classify_risk(
        self,
        fused_score: float
    ) -> Tuple[RiskLevel, DecisionType]:
        """
        Classify risk level and determine decision.
        
        High score = normal behavior = low risk → Allow
        Low score = anomalous = high risk → Deny
        
        Args:
            fused_score: Fused normalcy score [0.0, 1.0]
            
        Returns:
            Tuple of (RiskLevel, DecisionType)
        """
        if fused_score >= self.t1:
            # Low risk: score indicates normal behavior
            return RiskLevel.LOW, DecisionType.GRANT
        
        elif fused_score >= self.t2:
            # Medium risk: require additional authentication
            return RiskLevel.MEDIUM, DecisionType.REQUIRE_APPROVAL
        
        elif fused_score >= self.t3:
            # High risk: require manual review
            return RiskLevel.HIGH, DecisionType.REQUIRE_APPROVAL
        
        else:
            # Critical risk: deny access
            return RiskLevel.CRITICAL, DecisionType.DENY
    
    def _generate_reason(
        self,
        risk_level: RiskLevel,
        decision_type: DecisionType,
        fused_score: float,
        fusion_explanation: Dict
    ) -> str:
        """
        Generate human-readable reason for decision.
        
        Args:
            risk_level: Classified risk level
            decision_type: Decision type
            fused_score: Fused score
            fusion_explanation: Fusion layer explanation
            
        Returns:
            Reason string
        """
        if decision_type == DecisionType.GRANT:
            reason = (
                f"Access granted: Normal behavior detected "
                f"(score={fused_score:.3f}, risk=low)"
            )
        
        elif decision_type == DecisionType.REQUIRE_APPROVAL:
            if risk_level == RiskLevel.MEDIUM:
                reason = (
                    f"Additional authentication required: Moderate anomaly detected "
                    f"(score={fused_score:.3f}, risk=medium)"
                )
            else:  # HIGH
                reason = (
                    f"Manual review required: Significant anomaly detected "
                    f"(score={fused_score:.3f}, risk=high)"
                )
        
        elif decision_type == DecisionType.DENY:
            reason = (
                f"Access denied: Critical anomaly detected "
                f"(score={fused_score:.3f}, risk=critical)"
            )
        
        else:
            reason = f"Decision: {decision_type.value} (score={fused_score:.3f})"
        
        # Add fusion method info
        fusion_method = fusion_explanation.get('fusion_strategy', 'unknown')
        reason += f" [fusion={fusion_method}]"
        
        return reason
    
    def update_thresholds(
        self,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        t3: Optional[float] = None
    ):
        """
        Update decision thresholds dynamically.
        
        Args:
            t1: New T1 threshold (allow)
            t2: New T2 threshold (MFA)
            t3: New T3 threshold (review)
        """
        if t1 is not None:
            if not 0.0 <= t1 <= 1.0:
                raise ValueError(f"T1 must be in [0,1], got {t1}")
            self.t1 = t1
        
        if t2 is not None:
            if not 0.0 <= t2 <= 1.0:
                raise ValueError(f"T2 must be in [0,1], got {t2}")
            self.t2 = t2
        
        if t3 is not None:
            if not 0.0 <= t3 <= 1.0:
                raise ValueError(f"T3 must be in [0,1], got {t3}")
            self.t3 = t3
        
        # Validate ordering
        if not (self.t1 > self.t2 > self.t3):
            logger.warning(
                f"Thresholds not properly ordered: T1={self.t1}, T2={self.t2}, T3={self.t3}"
            )
        
        logger.info(f"Updated thresholds: T1={self.t1}, T2={self.t2}, T3={self.t3}")


class RBACEnforcement:
    """
    Final RBAC enforcement layer.
    
    Applies hard constraints regardless of ML/behavioral scores.
    Ensures critical safety rules are never violated.
    """
    
    def __init__(self):
        """Initialize RBAC enforcement."""
        self.rbac_config = config.get('rbac', {})
        self.enforcement_rules = self._load_enforcement_rules()
        
        logger.info(
            f"RBACEnforcement initialized with {len(self.enforcement_rules)} rules"
        )
    
    def _load_enforcement_rules(self) -> List[Dict]:
        """Load RBAC enforcement rules."""
        # Critical rules that override any other decision
        return [
            {
                'id': 'emergency_always_grant',
                'condition': lambda req, dec: req.emergency,
                'override': DecisionType.GRANT,
                'description': 'Emergency requests override all decisions',
            },
            {
                'id': 'auth_failure_always_deny',
                'condition': lambda req, dec: req.attempt_count > 5,
                'override': DecisionType.DENY,
                'description': 'Excessive auth failures always denied',
            },
            {
                'id': 'critical_action_human_required',
                'condition': lambda req, dec: (
                    req.agent_type.value == 'robot' and
                    str(req.action.value) in ['delete', 'override'] and
                    not req.human_present
                ),
                'override': DecisionType.DENY,
                'description': 'Critical robot actions require human supervision',
            },
        ]
    
    def enforce(
        self,
        request: AccessRequest,
        decision: AccessDecision
    ) -> AccessDecision:
        """
        Apply RBAC enforcement rules to decision.
        
        Args:
            request: Original access request
            decision: Decision from decision engine
            
        Returns:
            Potentially modified AccessDecision
        """
        original_decision = decision.decision
        
        # Check enforcement rules
        for rule in self.enforcement_rules:
            try:
                if rule['condition'](request, decision):
                    # Rule triggered - override decision
                    logger.warning(
                        f"RBAC override triggered: {rule['id']} - {rule['description']}"
                    )
                    
                    decision.decision = rule['override']
                    decision.reason = (
                        f"[RBAC OVERRIDE] {rule['description']}. "
                        f"Original: {original_decision.value} → "
                        f"Final: {rule['override'].value}"
                    )
                    
                    # Add to layer decisions
                    decision.layer_decisions['rbac_enforcement'] = {
                        'rule_id': rule['id'],
                        'description': rule['description'],
                        'original_decision': original_decision.value,
                        'override_decision': rule['override'].value,
                    }
                    
                    # Don't check further rules after first override
                    break
            
            except Exception as e:
                logger.error(f"Error evaluating RBAC rule {rule['id']}: {e}")
                continue
        
        return decision


class DecisionLogger:
    """
    Logs all access decisions for auditing and analysis.
    
    Supports:
    - Decision logging to file
    - Anomaly-specific logging
    - Statistics tracking
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize decision logger.
        
        Args:
            log_dir: Directory for decision logs
        """
        if log_dir is None:
            log_dir = Path(config.paths.get('logs_dir', 'logs'))
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.decisions_file = self.log_dir / 'decisions.jsonl'
        self.anomalies_file = self.log_dir / 'anomalies.jsonl'
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'grants': 0,
            'denials': 0,
            'approvals_required': 0,
            'rbac_overrides': 0,
        }
        
        logger.info(f"DecisionLogger initialized: {self.log_dir}")
    
    def log_decision(
        self,
        request: AccessRequest,
        decision: AccessDecision
    ):
        """
        Log an access decision.
        
        Args:
            request: Access request
            decision: Access decision
        """
        # Create log entry
        entry = {
            'timestamp': time.time(),
            'request_id': request.request_id,
            'agent_id': request.agent_id,
            'agent_type': request.agent_type.value,
            'action': request.action.value,
            'resource': request.resource,
            'decision': decision.decision.value,
            'confidence': decision.confidence,
            'latency_ms': decision.latency_ms,
            'reason': decision.reason,
        }
        
        # Log to file
        try:
            with open(self.decisions_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing decision log: {e}")
        
        # Update statistics
        self.stats['total_decisions'] += 1
        
        if decision.decision == DecisionType.GRANT:
            self.stats['grants'] += 1
        elif decision.decision == DecisionType.DENY:
            self.stats['denials'] += 1
        elif decision.decision == DecisionType.REQUIRE_APPROVAL:
            self.stats['approvals_required'] += 1
        
        if 'rbac_enforcement' in decision.layer_decisions:
            self.stats['rbac_overrides'] += 1
        
        # Log anomalies separately
        if decision.decision in [DecisionType.DENY, DecisionType.REQUIRE_APPROVAL]:
            self._log_anomaly(request, decision)
    
    def _log_anomaly(
        self,
        request: AccessRequest,
        decision: AccessDecision
    ):
        """
        Log anomalous decision to separate file.
        
        Args:
            request: Access request
            decision: Access decision
        """
        entry = {
            'timestamp': time.time(),
            'request_id': request.request_id,
            'agent_id': request.agent_id,
            'agent_type': request.agent_type.value,
            'action': request.action.value,
            'resource': request.resource,
            'location': request.location,
            'human_present': request.human_present,
            'emergency': request.emergency,
            'decision': decision.decision.value,
            'confidence': decision.confidence,
            'reason': decision.reason,
            'layer_decisions': decision.layer_decisions,
        }
        
        try:
            with open(self.anomalies_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing anomaly log: {e}")
    
    def get_statistics(self) -> Dict:
        """Get decision statistics."""
        stats = self.stats.copy()
        
        # Calculate rates
        total = stats['total_decisions']
        if total > 0:
            stats['grant_rate'] = stats['grants'] / total
            stats['denial_rate'] = stats['denials'] / total
            stats['approval_rate'] = stats['approvals_required'] / total
            stats['override_rate'] = stats['rbac_overrides'] / total
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Decision statistics reset")


class DecisionLayer:
    """Orchestrates the complete decision pipeline."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize decision layer.
        
        Args:
            log_dir: Optional log directory
        """
        self.decision_engine = DecisionEngine()
        self.rbac_enforcement = RBACEnforcement()
        self.decision_logger = DecisionLogger(log_dir)
        
        logger.info("DecisionLayer initialized")
    
    def make_decision(
        self,
        request: AccessRequest,
        fused_result: Dict
    ) -> AccessDecision:
        """
        Make final access control decision.
        
        Pipeline:
        1. Decision Engine: Classify risk and decide
        2. RBAC Enforcement: Apply hard constraints
        3. Decision Logger: Log decision
        
        Args:
            request: Access request
            fused_result: Fused result from fusion layer
            
        Returns:
            Final AccessDecision
        """
        # Extract fusion results
        fused_score = fused_result['confidence']  # Use confidence as score
        fused_confidence = fused_result['confidence']
        fusion_explanation = fused_result['explanation']
        
        # Step 1: Decision Engine
        decision = self.decision_engine.decide(
            fused_score=fused_score,
            fused_confidence=fused_confidence,
            request=request,
            fusion_explanation=fusion_explanation
        )
        
        # Step 2: RBAC Enforcement
        decision = self.rbac_enforcement.enforce(request, decision)
        
        # Step 3: Log Decision
        self.decision_logger.log_decision(request, decision)
        
        logger.info(
            f"Final decision for {request.request_id}: {decision.decision.value} "
            f"(confidence={decision.confidence:.3f}, latency={decision.latency_ms:.2f}ms)"
        )
        
        return decision
    
    def get_statistics(self) -> Dict:
        """Get decision statistics."""
        return self.decision_logger.get_statistics()
    
    def update_thresholds(
        self,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        t3: Optional[float] = None
    ):
        """Update decision thresholds."""
        self.decision_engine.update_thresholds(t1, t2, t3)