#!/usr/bin/env python3
"""
BBAC ICS Framework - Continuous Learning Layer

Implements:
- Updater (model and profile updates)
- Anomaly Logger (anomaly tracking and analysis)
- Profile Update (with trust filter)

From flowchart:
- Trust gate: Only update if trusted (score > threshold)
- Trusted buffer: Accumulate trusted samples for retraining
- Quarantine: Isolate untrusted samples
- Dynamic rules: Update rules based on patterns
"""

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..util.config_loader import config
from ..util.data_structures import (
    AccessRequest,
    AccessDecision,
    DecisionType,
)


logger = logging.getLogger(__name__)


class TrustFilter:
    """
    Trust filter for continuous learning.
    
    Implements trust gate from flowchart:
    - Only updates profiles/models with trusted data
    - Prevents learning from anomalous behavior
    """
    
    def __init__(self):
        """Initialize trust filter with thresholds."""
        self.thresholds_config = config.thresholds
        
        # Trust threshold from config
        self.trust_threshold = self.thresholds_config.get('trust_threshold', 0.8)
        
        # Additional criteria
        self.min_confidence = self.thresholds_config.get('min_confidence_for_update', 0.7)
        self.require_grant = self.thresholds_config.get('require_grant_for_update', True)
        
        logger.info(
            f"TrustFilter initialized: threshold={self.trust_threshold}, "
            f"min_confidence={self.min_confidence}"
        )
    
    def is_trusted(
        self,
        request: AccessRequest,
        decision: AccessDecision,
        fused_score: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if request/decision pair is trusted for learning.
        
        Args:
            request: Access request
            decision: Access decision
            fused_score: Optional fused normalcy score
            
        Returns:
            Tuple of (is_trusted, reason)
        """
        reasons = []
        
        # Criterion 1: Decision confidence must be high
        if decision.confidence < self.min_confidence:
            return False, f"Low confidence: {decision.confidence:.3f} < {self.min_confidence}"
        
        # Criterion 2: Must be GRANT decision (if required)
        if self.require_grant and decision.decision != DecisionType.GRANT:
            return False, f"Non-grant decision: {decision.decision.value}"
        
        # Criterion 3: Fused score must be above threshold (if provided)
        if fused_score is not None:
            if fused_score < self.trust_threshold:
                return False, f"Low trust score: {fused_score:.3f} < {self.trust_threshold}"
        
        # Criterion 4: No emergency override
        if request.emergency:
            return False, "Emergency request - not representative of normal behavior"
        
        # Criterion 5: No RBAC override
        if 'rbac_enforcement' in decision.layer_decisions:
            return False, "RBAC override present - not representative"
        
        # Criterion 6: Not too many auth failures
        if request.attempt_count > 2:
            return False, f"High auth attempts: {request.attempt_count}"
        
        # All criteria passed
        return True, "All trust criteria satisfied"
    
    def update_threshold(self, new_threshold: float):
        """Update trust threshold dynamically."""
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0,1], got {new_threshold}")
        
        self.trust_threshold = new_threshold
        logger.info(f"Updated trust threshold: {new_threshold}")


class TrustedBuffer:
    """
    Buffer for accumulating trusted samples before model retraining.
    
    Implements trusted_buffer from flowchart.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize trusted buffer.
        
        Args:
            max_size: Maximum buffer size before triggering retrain
        """
        if max_size is None:
            max_size = config.get('learning', {}).get('buffer_size', 1000)
        
        self.max_size = max_size
        self.buffer: List[Dict] = []
        
        # Statistics
        self.total_added = 0
        self.total_flushed = 0
        
        logger.info(f"TrustedBuffer initialized: max_size={max_size}")
    
    def add(
        self,
        request: AccessRequest,
        decision: AccessDecision,
        fused_score: float
    ):
        """
        Add trusted sample to buffer.
        
        Args:
            request: Access request
            decision: Access decision
            fused_score: Fused score
        """
        sample = {
            'timestamp': time.time(),
            'agent_id': request.agent_id,
            'agent_type': request.agent_type.value,
            'agent_role': request.agent_role.value,
            'action': request.action.value,
            'resource': request.resource,
            'resource_type': request.resource_type.value,
            'location': request.location,
            'human_present': request.human_present,
            'emergency': request.emergency,
            'previous_action': request.previous_action.value if request.previous_action else None,
            'decision': decision.decision.value,
            'confidence': decision.confidence,
            'fused_score': fused_score,
        }
        
        self.buffer.append(sample)
        self.total_added += 1
        
        logger.debug(f"Added to buffer: {len(self.buffer)}/{self.max_size}")
    
    def is_full(self) -> bool:
        """Check if buffer is full and ready for retraining."""
        return len(self.buffer) >= self.max_size
    
    def get_samples(self, agent_id: Optional[str] = None) -> List[Dict]:
        """
        Get samples from buffer, optionally filtered by agent.
        
        Args:
            agent_id: Optional agent filter
            
        Returns:
            List of samples
        """
        if agent_id is None:
            return self.buffer.copy()
        
        return [s for s in self.buffer if s['agent_id'] == agent_id]
    
    def to_dataframe(self, agent_id: Optional[str] = None) -> pd.DataFrame:
        """
        Convert buffer to DataFrame.
        
        Args:
            agent_id: Optional agent filter
            
        Returns:
            DataFrame with buffer samples
        """
        samples = self.get_samples(agent_id)
        
        if not samples:
            return pd.DataFrame()
        
        return pd.DataFrame(samples)
    
    def flush(self, agent_id: Optional[str] = None) -> int:
        """
        Flush buffer (clear samples).
        
        Args:
            agent_id: Optional agent filter (only flush specific agent)
            
        Returns:
            Number of samples flushed
        """
        if agent_id is None:
            # Flush all
            count = len(self.buffer)
            self.buffer.clear()
        else:
            # Flush only specific agent
            count = sum(1 for s in self.buffer if s['agent_id'] == agent_id)
            self.buffer = [s for s in self.buffer if s['agent_id'] != agent_id]
        
        self.total_flushed += count
        logger.info(f"Flushed {count} samples from buffer")
        
        return count
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        return {
            'current_size': len(self.buffer),
            'max_size': self.max_size,
            'fill_percentage': (len(self.buffer) / self.max_size * 100) if self.max_size > 0 else 0,
            'total_added': self.total_added,
            'total_flushed': self.total_flushed,
            'is_full': self.is_full(),
        }


class QuarantineManager:
    """
    Manages quarantined (untrusted) samples.
    
    Implements quarantine from flowchart.
    """
    
    def __init__(self, quarantine_dir: Optional[Path] = None):
        """
        Initialize quarantine manager.
        
        Args:
            quarantine_dir: Directory for quarantined samples
        """
        if quarantine_dir is None:
            quarantine_dir = Path(config.paths.get('quarantine_dir', 'quarantine'))
        
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.quarantine_file = self.quarantine_dir / 'quarantined_samples.jsonl'
        
        # In-memory cache
        self.quarantine_samples: List[Dict] = []
        
        # Statistics
        self.total_quarantined = 0
        
        logger.info(f"QuarantineManager initialized: {self.quarantine_dir}")
    
    def quarantine(
        self,
        request: AccessRequest,
        decision: AccessDecision,
        reason: str
    ):
        """
        Quarantine an untrusted sample.
        
        Args:
            request: Access request
            decision: Access decision
            reason: Reason for quarantine
        """
        sample = {
            'timestamp': time.time(),
            'request_id': request.request_id,
            'agent_id': request.agent_id,
            'agent_type': request.agent_type.value,
            'action': request.action.value,
            'resource': request.resource,
            'location': request.location,
            'decision': decision.decision.value,
            'confidence': decision.confidence,
            'quarantine_reason': reason,
            'layer_decisions': decision.layer_decisions,
        }
        
        # Add to memory
        self.quarantine_samples.append(sample)
        self.total_quarantined += 1
        
        # Write to disk
        try:
            with open(self.quarantine_file, 'a') as f:
                f.write(json.dumps(sample) + '\n')
        except Exception as e:
            logger.error(f"Error writing quarantine file: {e}")
        
        logger.debug(f"Quarantined sample: {request.request_id} - {reason}")
    
    def get_samples(
        self,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get quarantined samples.
        
        Args:
            agent_id: Optional agent filter
            limit: Optional limit on number of samples
            
        Returns:
            List of quarantined samples
        """
        samples = self.quarantine_samples
        
        if agent_id:
            samples = [s for s in samples if s['agent_id'] == agent_id]
        
        if limit:
            samples = samples[-limit:]
        
        return samples
    
    def get_statistics(self) -> Dict:
        """Get quarantine statistics."""
        stats = {
            'total_quarantined': self.total_quarantined,
            'current_count': len(self.quarantine_samples),
        }
        
        # Count by agent
        if self.quarantine_samples:
            agent_counts = defaultdict(int)
            for sample in self.quarantine_samples:
                agent_counts[sample['agent_id']] += 1
            
            stats['by_agent'] = dict(agent_counts)
        
        return stats


class ProfileUpdater:
    """
    Updates agent behavioral profiles with trusted data.
    
    Integrates with ProfileManager from modeling layer.
    """
    
    def __init__(self, profile_manager):
        """
        Initialize profile updater.
        
        Args:
            profile_manager: ProfileManager instance from modeling layer
        """
        self.profile_manager = profile_manager
        
        self.update_config = config.get('learning', {})
        self.min_samples_for_update = self.update_config.get('min_samples_for_update', 100)
        
        # Track updates
        self.update_history = defaultdict(list)
        
        logger.info(
            f"ProfileUpdater initialized: min_samples={self.min_samples_for_update}"
        )
    
    def update_profile(
        self,
        agent_id: str,
        trusted_data: pd.DataFrame,
        trust_score: float
    ) -> bool:
        """
        Update agent profile with trusted data.
        
        Args:
            agent_id: Agent identifier
            trusted_data: DataFrame with trusted samples
            trust_score: Average trust score
            
        Returns:
            True if updated successfully
        """
        # Check minimum samples
        if len(trusted_data) < self.min_samples_for_update:
            logger.debug(
                f"Insufficient samples for {agent_id}: "
                f"{len(trusted_data)} < {self.min_samples_for_update}"
            )
            return False
        
        # Check if profile exists
        if not self.profile_manager.profile_exists(agent_id):
            logger.warning(f"Profile not found for {agent_id}, cannot update")
            return False
        
        # Build new baseline from trusted data
        # (this requires BaselineBuilder from modeling layer)
        from ..core.modeling import BaselineBuilder
        
        baseline_builder = BaselineBuilder()
        new_baseline = baseline_builder.build_baseline(trusted_data)
        
        # Update profile through ProfileManager (with trust filter)
        success = self.profile_manager.update_profile(
            agent_id=agent_id,
            new_baseline=new_baseline,
            trust_score=trust_score
        )
        
        if success:
            # Record update
            self.update_history[agent_id].append({
                'timestamp': time.time(),
                'sample_count': len(trusted_data),
                'trust_score': trust_score,
            })
            
            logger.info(
                f"Updated profile for {agent_id}: "
                f"{len(trusted_data)} samples, trust={trust_score:.3f}"
            )
        
        return success
    
    def get_update_history(self, agent_id: str) -> List[Dict]:
        """Get update history for agent."""
        return self.update_history.get(agent_id, [])


class ModelUpdater:
    """
    Updates ML models with trusted data.
    
    Placeholder for future ML model retraining.
    """
    
    def __init__(self):
        """Initialize model updater."""
        self.update_config = config.get('learning', {})
        
        # Track retraining
        self.retrain_history = []
        
        logger.info("ModelUpdater initialized (placeholder)")
    
    def retrain_models(
        self,
        trusted_data: pd.DataFrame,
        agent_id: Optional[str] = None
    ) -> bool:
        """
        Retrain ML models with trusted data.
        
        Args:
            trusted_data: DataFrame with trusted samples
            agent_id: Optional agent filter
            
        Returns:
            True if retrained successfully
        """
        # TODO: Implement actual model retraining
        # - Isolation Forest retraining
        # - LSTM/Sequence model retraining
        # - Feature scaler updates
        
        logger.warning("ModelUpdater.retrain_models() not implemented yet")
        
        # Placeholder: just record the attempt
        self.retrain_history.append({
            'timestamp': time.time(),
            'sample_count': len(trusted_data),
            'agent_id': agent_id,
        })
        
        return False  # Not actually retrained
    
    def get_retrain_history(self) -> List[Dict]:
        """Get retraining history."""
        return self.retrain_history


class AnomalyLogger:
    """
    Specialized logger for anomaly analysis.
    
    Tracks patterns in anomalies for dynamic rule generation.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize anomaly logger.
        
        Args:
            log_dir: Directory for anomaly logs
        """
        if log_dir is None:
            log_dir = Path(config.paths.get('logs_dir', 'logs'))
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.anomaly_patterns_file = self.log_dir / 'anomaly_patterns.json'
        
        # In-memory tracking
        self.anomaly_counts = defaultdict(int)
        self.anomaly_patterns = defaultdict(list)
        
        logger.info(f"AnomalyLogger initialized: {self.log_dir}")
    
    def log_anomaly(
        self,
        request: AccessRequest,
        decision: AccessDecision,
        fused_score: float
    ):
        """
        Log an anomaly for pattern analysis.
        
        Args:
            request: Access request
            decision: Access decision
            fused_score: Fused score
        """
        # Create pattern key
        pattern_key = self._create_pattern_key(request, decision)
        
        # Increment count
        self.anomaly_counts[pattern_key] += 1
        
        # Store sample
        self.anomaly_patterns[pattern_key].append({
            'timestamp': time.time(),
            'request_id': request.request_id,
            'agent_id': request.agent_id,
            'fused_score': fused_score,
            'confidence': decision.confidence,
        })
        
        logger.debug(f"Logged anomaly pattern: {pattern_key}")
    
    def _create_pattern_key(
        self,
        request: AccessRequest,
        decision: AccessDecision
    ) -> str:
        """Create pattern key for anomaly categorization."""
        # Combine key features
        components = [
            request.agent_type.value,
            request.action.value,
            request.resource_type.value,
            'emergency' if request.emergency else 'normal',
            'human_present' if request.human_present else 'no_human',
            decision.decision.value,
        ]
        
        return '|'.join(components)
    
    def get_frequent_patterns(self, min_count: int = 5) -> List[Tuple[str, int]]:
        """
        Get frequently occurring anomaly patterns.
        
        Args:
            min_count: Minimum occurrence count
            
        Returns:
            List of (pattern, count) tuples
        """
        frequent = [
            (pattern, count)
            for pattern, count in self.anomaly_counts.items()
            if count >= min_count
        ]
        
        # Sort by count descending
        frequent.sort(key=lambda x: x[1], reverse=True)
        
        return frequent
    
    def suggest_new_rules(self, min_frequency: int = 10) -> List[Dict]:
        """
        Suggest new rules based on frequent anomaly patterns.
        
        Args:
            min_frequency: Minimum pattern frequency
            
        Returns:
            List of suggested rule dictionaries
        """
        suggestions = []
        
        frequent_patterns = self.get_frequent_patterns(min_count=min_frequency)
        
        for pattern, count in frequent_patterns:
            # Parse pattern
            components = pattern.split('|')
            
            if len(components) >= 6:
                agent_type, action, resource_type, emergency, human, decision = components[:6]
                
                suggestion = {
                    'pattern': pattern,
                    'frequency': count,
                    'suggested_rule': {
                        'agent_type': agent_type,
                        'action': action,
                        'resource_type': resource_type,
                        'emergency': emergency == 'emergency',
                        'human_present': human == 'human_present',
                        'recommended_decision': decision,
                    },
                    'confidence': min(1.0, count / 100),  # Simple heuristic
                }
                
                suggestions.append(suggestion)
        
        logger.info(f"Generated {len(suggestions)} rule suggestions")
        return suggestions
    
    def save_patterns(self):
        """Save anomaly patterns to disk."""
        patterns_data = {
            'counts': dict(self.anomaly_counts),
            'timestamp': time.time(),
        }
        
        try:
            with open(self.anomaly_patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            logger.info(f"Saved anomaly patterns to {self.anomaly_patterns_file}")
        except Exception as e:
            logger.error(f"Error saving anomaly patterns: {e}")


class ContinuousLearningLayer:
    """Orchestrates the continuous learning pipeline."""
    
    def __init__(
        self,
        profile_manager,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize continuous learning layer.
        
        Args:
            profile_manager: ProfileManager instance from modeling layer
            log_dir: Optional log directory
        """
        self.trust_filter = TrustFilter()
        self.trusted_buffer = TrustedBuffer()
        self.quarantine_manager = QuarantineManager()
        self.profile_updater = ProfileUpdater(profile_manager)
        self.model_updater = ModelUpdater()
        self.anomaly_logger = AnomalyLogger(log_dir)
        
        logger.info("ContinuousLearningLayer initialized")
    
    def process_decision(
        self,
        request: AccessRequest,
        decision: AccessDecision,
        fused_score: float
    ):
        """
        Process decision for continuous learning.
        
        Pipeline:
        1. Trust filter: Determine if trusted
        2a. If trusted: Add to buffer
        2b. If untrusted: Quarantine
        3. Log anomaly if applicable
        4. Check buffer and trigger updates
        
        Args:
            request: Access request
            decision: Access decision
            fused_score: Fused normalcy score
        """
        # Step 1: Trust filter
        is_trusted, reason = self.trust_filter.is_trusted(request, decision, fused_score)
        
        if is_trusted:
            # Step 2a: Add to trusted buffer
            self.trusted_buffer.add(request, decision, fused_score)
            
            logger.debug(f"Added to trusted buffer: {request.request_id}")
            
            # Step 4: Check if buffer is full
            if self.trusted_buffer.is_full():
                self._trigger_updates()
        
        else:
            # Step 2b: Quarantine
            self.quarantine_manager.quarantine(request, decision, reason)
            
            logger.debug(f"Quarantined: {request.request_id} - {reason}")
        
        # Step 3: Log anomalies
        if decision.decision in [DecisionType.DENY, DecisionType.REQUIRE_APPROVAL]:
            self.anomaly_logger.log_anomaly(request, decision, fused_score)
    
    def _trigger_updates(self):
        """Trigger profile and model updates when buffer is full."""
        logger.info("Buffer full - triggering updates")
        
        # Get buffer data
        buffer_df = self.trusted_buffer.to_dataframe()
        
        if buffer_df.empty:
            logger.warning("Buffer is empty, cannot update")
            return
        
        # Group by agent
        agents = buffer_df['agent_id'].unique()
        
        for agent_id in agents:
            agent_data = buffer_df[buffer_df['agent_id'] == agent_id]
            
            # Calculate average trust score
            avg_trust = agent_data['fused_score'].mean()
            
            # Update profile
            self.profile_updater.update_profile(
                agent_id=agent_id,
                trusted_data=agent_data,
                trust_score=avg_trust
            )
        
        # Retrain models (if implemented)
        self.model_updater.retrain_models(buffer_df)
        
        # Flush buffer after updates
        self.trusted_buffer.flush()
        
        logger.info("Updates complete, buffer flushed")
    
    def force_update(self, agent_id: Optional[str] = None):
        """
        Force immediate update without waiting for buffer to fill.
        
        Args:
            agent_id: Optional agent filter
        """
        logger.info(f"Forcing update for {agent_id if agent_id else 'all agents'}")
        
        buffer_df = self.trusted_buffer.to_dataframe(agent_id)
        
        if buffer_df.empty:
            logger.warning("No data in buffer for update")
            return
        
        if agent_id:
            # Update specific agent
            avg_trust = buffer_df['fused_score'].mean()
            self.profile_updater.update_profile(agent_id, buffer_df, avg_trust)
            self.trusted_buffer.flush(agent_id)
        else:
            # Update all agents
            self._trigger_updates()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive learning statistics."""
        return {
            'trusted_buffer': self.trusted_buffer.get_statistics(),
            'quarantine': self.quarantine_manager.get_statistics(),
            'frequent_anomaly_patterns': len(self.anomaly_logger.get_frequent_patterns()),
        }
    
    def get_rule_suggestions(self, min_frequency: int = 10) -> List[Dict]:
        """Get suggested rules based on anomaly patterns."""
        return self.anomaly_logger.suggest_new_rules(min_frequency)