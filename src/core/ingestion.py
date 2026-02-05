#!/usr/bin/env python3
"""
BBAC ICS Framework - Ingestion Layer

Implements:
- Authentication module simulation
- Log collection and filtering
- Data preprocessing for analysis
- Integration with dataset
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from data.loader import DatasetLoader
from ..util.config_loader import config
from ..util.data_structures import (
    AccessRequest,
    AgentID,
    AuthStatus,
    ActionType,
)


logger = logging.getLogger(__name__)


class AuthenticationModule:
    """Simulates authentication module (from flowchart)."""
    
    def __init__(self):
        self.thresholds = config.thresholds
        self.max_attempts = self.thresholds.get('max_auth_attempts', 3)
        self.failed_attempts = {}
    
    def authenticate(self, request: AccessRequest) -> bool:
        """
        Check authentication status from request.
        
        Args:
            request: AccessRequest with auth_status field
            
        Returns:
            True if authentication successful
        """
        agent_id = request.agent_id
        
        # Check auth_status from dataset
        if request.auth_status == AuthStatus.FAILED:
            # Track failed attempts
            if agent_id not in self.failed_attempts:
                self.failed_attempts[agent_id] = 0
            self.failed_attempts[agent_id] += 1
            
            logger.debug(f"Auth failed for {agent_id} (attempt {self.failed_attempts[agent_id]})")
            return False
        
        # Check if exceeded max attempts (from dataset attempt_count or internal tracking)
        attempt_count = max(request.attempt_count, self.failed_attempts.get(agent_id, 0))
        
        if attempt_count > self.max_attempts:
            logger.warning(f"Max auth attempts exceeded for {agent_id}: {attempt_count}")
            return False
        
        # Reset counter on success
        if agent_id in self.failed_attempts:
            self.failed_attempts[agent_id] = 0
        
        return True
    
    def reset_attempts(self, agent_id: AgentID):
        """Reset failed attempt counter for agent."""
        if agent_id in self.failed_attempts:
            del self.failed_attempts[agent_id]


class LogCollector:
    """Collects and filters access logs (from flowchart START)."""
    
    def __init__(self):
        self.baseline_config = config.baseline
        self.window_days = self.baseline_config.get('window_days', 10)
        self.recent_weight = self.baseline_config.get('recent_weight', 0.7)
        self.historical_weight = 1.0 - self.recent_weight
        
        logger.info(f"LogCollector: window={self.window_days}d, recent_weight={self.recent_weight}")
    
    def collect_logs(self, dataset_loader: DatasetLoader, split: str = 'train') -> pd.DataFrame:
        """
        Collect logs from dataset for baseline calculation.
        
        Args:
            dataset_loader: DatasetLoader instance
            split: Dataset split to use ('train', 'validation', 'test')
            
        Returns:
            DataFrame with logs
        """
        if split == 'train':
            data = dataset_loader.train_data
        elif split == 'validation':
            data = dataset_loader.validation_data
        elif split == 'test':
            data = dataset_loader.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if data is None or data.empty:
            logger.warning(f"No data available for split '{split}'")
            return pd.DataFrame()
        
        logger.info(f"Collected {len(data)} logs from {split} split")
        return data.copy()
    
    def filter_for_agent(self, logs: pd.DataFrame, agent_id: str) -> pd.DataFrame:
        """
        Filter logs for a single agent and sort by timestamp.
        
        Args:
            logs: Complete log DataFrame
            agent_id: Target agent identifier
            
        Returns:
            Filtered and sorted logs
        """
        # Handle both 'agent_id' and 'user_id' columns
        agent_col = 'agent_id' if 'agent_id' in logs.columns else 'user_id'
        
        filtered = logs[logs[agent_col] == agent_id].copy()
        
        if 'timestamp' in filtered.columns:
            filtered = filtered.sort_values('timestamp')
        
        logger.debug(f"Filtered {len(filtered)} logs for agent {agent_id}")
        return filtered
    
    def apply_sliding_window(
        self, 
        logs: pd.DataFrame, 
        reference_time: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Apply sliding window to logs (adaptive baseline).
        
        Implements: Baseline(t) = 70% recent + 30% historical
        
        Args:
            logs: Historical logs
            reference_time: Reference timestamp (default: most recent)
            
        Returns:
            Windowed logs with weights
        """
        if logs.empty or 'timestamp' not in logs.columns:
            return logs
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(logs['timestamp']):
            logs['timestamp'] = pd.to_datetime(logs['timestamp'])
        
        # Determine reference time
        if reference_time is None:
            reference_time = logs['timestamp'].max()
        
        # Calculate window boundaries
        window_start = reference_time - pd.Timedelta(days=self.window_days)
        recent_boundary = reference_time - pd.Timedelta(days=self.window_days // 2)
        
        # Filter to window
        windowed = logs[logs['timestamp'] >= window_start].copy()
        
        if windowed.empty:
            logger.warning("No logs within sliding window")
            return windowed
        
        # Assign weights: 70% recent, 30% historical
        windowed['baseline_weight'] = np.where(
            windowed['timestamp'] >= recent_boundary,
            self.recent_weight / len(windowed[windowed['timestamp'] >= recent_boundary]),
            self.historical_weight / len(windowed[windowed['timestamp'] < recent_boundary])
        )
        
        logger.debug(
            f"Sliding window: {len(windowed)} logs "
            f"({(windowed['timestamp'] >= recent_boundary).sum()} recent, "
            f"{(windowed['timestamp'] < recent_boundary).sum()} historical)"
        )
        
        return windowed


class DataPreprocessor:
    """Preprocesses data for modeling and analysis."""
    
    def __init__(self):
        self.ml_config = config.ml_params
        self.feature_cache = {}
        
    def preprocess_request(self, request: AccessRequest) -> AccessRequest:
        """
        Validate and enrich AccessRequest.
        
        Args:
            request: AccessRequest from dataset
            
        Returns:
            Validated AccessRequest
        """
        # Already structured from DatasetLoader.to_access_request()
        # Just validate critical fields
        
        if not request.agent_id:
            raise ValueError("AccessRequest missing agent_id")
        
        if request.timestamp <= 0:
            logger.warning(f"Invalid timestamp for request {request.request_id}, using current time")
            import time
            request.timestamp = time.time()
        
        return request
    
    def calculate_features(
        self, 
        historical_data: pd.DataFrame,
        agent_id: Optional[str] = None
    ) -> Dict:
        """
        Calculate features from historical data (from flowchart FEATURE EXTRACTION).
        
        Args:
            historical_data: Historical access logs
            agent_id: Optional agent filter for caching
            
        Returns:
            Dictionary of extracted features
        """
        # Check cache
        cache_key = f"{agent_id}_{len(historical_data)}" if agent_id else f"all_{len(historical_data)}"
        if cache_key in self.feature_cache:
            logger.debug(f"Using cached features for {cache_key}")
            return self.feature_cache[cache_key]
        
        features = {
            'temporal_features': self._extract_temporal_features(historical_data),
            'frequency_features': self._extract_frequency_features(historical_data),
            'sequence_features': self._extract_sequence_features(historical_data),
            'context_features': self._extract_context_features(historical_data),
        }
        
        # Cache result
        self.feature_cache[cache_key] = features
        
        logger.debug(f"Calculated features for {len(historical_data)} records")
        return features
    
    def _extract_temporal_features(self, data: pd.DataFrame) -> Dict:
        """Extract temporal features (hour, gap, sequence position)."""
        if 'timestamp' not in data.columns or data.empty:
            return {}
        
        data = data.sort_values('timestamp')
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            timestamps = data['timestamp']
        
        features = {
            'hour_of_day_distribution': timestamps.dt.hour.value_counts().to_dict(),
            'day_of_week_distribution': timestamps.dt.dayofweek.value_counts().to_dict(),
            'avg_time_gap': self._calculate_avg_time_gap(timestamps),
            'time_gap_std': self._calculate_time_gap_std(timestamps),
            'hourly_pattern': timestamps.dt.hour.value_counts().sort_index().to_dict(),
            'peak_hours': self._identify_peak_hours(timestamps),
        }
        
        return features
    
    def _calculate_avg_time_gap(self, timestamps: pd.Series) -> float:
        """Calculate average time between requests."""
        if len(timestamps) < 2:
            return 0.0
        diffs = timestamps.diff().dt.total_seconds().dropna()
        return float(diffs.mean()) if len(diffs) > 0 else 0.0
    
    def _calculate_time_gap_std(self, timestamps: pd.Series) -> float:
        """Calculate standard deviation of time gaps."""
        if len(timestamps) < 2:
            return 0.0
        diffs = timestamps.diff().dt.total_seconds().dropna()
        return float(diffs.std()) if len(diffs) > 0 else 0.0
    
    def _identify_peak_hours(self, timestamps: pd.Series, top_n: int = 3) -> List[int]:
        """Identify peak usage hours."""
        hourly_counts = timestamps.dt.hour.value_counts()
        return hourly_counts.nlargest(top_n).index.tolist()
    
    def _extract_frequency_features(self, data: pd.DataFrame) -> Dict:
        """Extract frequency features."""
        if data.empty:
            return {}
        
        features = {
            'total_accesses': len(data),
        }
        
        # Count unique values for each column
        for col in ['resource', 'action', 'location']:
            if col in data.columns:
                features[f'unique_{col}s'] = int(data[col].nunique())
                features[f'{col}_distribution'] = data[col].value_counts().to_dict()
        
        # Emergency frequency
        if 'emergency_flag' in data.columns:
            features['emergency_count'] = int(data['emergency_flag'].sum())
            features['emergency_rate'] = float(data['emergency_flag'].mean())
        
        # Human presence
        if 'human_present' in data.columns:
            features['human_present_count'] = int(data['human_present'].sum())
            features['human_present_rate'] = float(data['human_present'].mean())
        
        return features
    
    def _extract_sequence_features(self, data: pd.DataFrame) -> Dict:
        """Extract sequence features (for Markov analysis)."""
        if len(data) < 2 or 'action' not in data.columns:
            return {}
        
        # Sort by timestamp if available
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        # Build action sequences
        actions = data['action'].tolist()
        sequences = []
        
        for i in range(len(actions) - 1):
            seq = f"{actions[i]}->{actions[i+1]}"
            sequences.append(seq)
        
        features = {
            'common_sequences': pd.Series(sequences).value_counts().head(10).to_dict(),
            'sequence_length': len(sequences),
            'sequence_entropy': self._calculate_entropy(sequences),
            'unique_transitions': len(set(sequences)),
        }
        
        # Action repetition rate
        if len(actions) > 1:
            repetitions = sum(1 for i in range(len(actions)-1) if actions[i] == actions[i+1])
            features['action_repetition_rate'] = repetitions / (len(actions) - 1)
        
        return features
    
    def _extract_context_features(self, data: pd.DataFrame) -> Dict:
        """Extract contextual features (location, resource_type, etc)."""
        if data.empty:
            return {}
        
        features = {}
        
        # Location patterns
        if 'location' in data.columns:
            features['location_diversity'] = int(data['location'].nunique())
            features['most_common_location'] = data['location'].mode()[0] if len(data) > 0 else None
        
        # Resource type patterns
        if 'resource_type' in data.columns:
            features['resource_type_distribution'] = data['resource_type'].value_counts().to_dict()
        
        # Agent type distribution (for mixed agent scenarios)
        if 'agent_type' in data.columns:
            features['agent_type_distribution'] = data['agent_type'].value_counts().to_dict()
        
        return features
    
    def _calculate_entropy(self, items: List) -> float:
        """Calculate entropy of a list of items."""
        from collections import Counter
        import math
        
        if not items:
            return 0.0
        
        counter = Counter(items)
        total = len(items)
        entropy = 0.0
        
        for count in counter.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        
        return float(entropy)
    
    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
        logger.debug("Feature cache cleared")


class IngestionLayer:
    """Orchestrates the complete ingestion pipeline."""
    
    def __init__(self, dataset_loader: Optional[DatasetLoader] = None):
        """
        Initialize ingestion layer.
        
        Args:
            dataset_loader: Optional DatasetLoader instance
        """
        self.auth = AuthenticationModule()
        self.log_collector = LogCollector()
        self.preprocessor = DataPreprocessor()
        
        # Initialize or accept dataset loader
        if dataset_loader is None:
            self.dataset_loader = DatasetLoader()
            self.dataset_loader.load_all()
        else:
            self.dataset_loader = dataset_loader
        
        logger.info("IngestionLayer initialized")
    
    def process_request(self, request: AccessRequest) -> Optional[AccessRequest]:
        """
        Process request through ingestion pipeline.
        
        Args:
            request: AccessRequest (already structured from dataset)
            
        Returns:
            Processed AccessRequest or None if authentication fails
        """
        # Step 1: Authentication check
        if not self.auth.authenticate(request):
            logger.warning(
                f"Authentication failed for agent {request.agent_id} "
                f"(status: {request.auth_status}, attempts: {request.attempt_count})"
            )
            return None
        
        # Step 2: Preprocess/validate
        try:
            processed = self.preprocessor.preprocess_request(request)
        except ValueError as e:
            logger.error(f"Preprocessing failed for request {request.request_id}: {e}")
            return None
        
        logger.debug(f"Processed request: {processed.request_id}")
        return processed
    
    def prepare_baseline_data(
        self, 
        agent_id: Optional[str] = None,
        split: str = 'train',
        apply_window: bool = True
    ) -> pd.DataFrame:
        """
        Prepare baseline data from dataset.
        
        Args:
            agent_id: Optional agent filter
            split: Dataset split to use
            apply_window: Whether to apply sliding window
            
        Returns:
            Prepared DataFrame for baseline calculation
        """
        # Collect logs from dataset
        logs = self.log_collector.collect_logs(self.dataset_loader, split=split)
        
        if logs.empty:
            logger.warning("No logs available for baseline")
            return logs
        
        # Filter for specific agent if requested
        if agent_id:
            logs = self.log_collector.filter_for_agent(logs, agent_id)
        
        # Apply sliding window for adaptive baseline
        if apply_window:
            logs = self.log_collector.apply_sliding_window(logs)
        
        # Clean data
        logs = self._clean_log_data(logs)
        
        logger.info(f"Prepared {len(logs)} baseline records" + 
                   (f" for agent {agent_id}" if agent_id else ""))
        
        return logs
    
    def extract_features_for_agent(
        self, 
        agent_id: str,
        split: str = 'train'
    ) -> Dict:
        """
        Extract features for specific agent from dataset.
        
        Args:
            agent_id: Agent identifier
            split: Dataset split to use
            
        Returns:
            Feature dictionary
        """
        baseline_data = self.prepare_baseline_data(agent_id=agent_id, split=split)
        
        if baseline_data.empty:
            logger.warning(f"No baseline data for agent {agent_id}")
            return {}
        
        features = self.preprocessor.calculate_features(baseline_data, agent_id=agent_id)
        
        return features
    
    def batch_process_requests(
        self, 
        requests: List[AccessRequest]
    ) -> List[AccessRequest]:
        """
        Process multiple requests in batch.
        
        Args:
            requests: List of AccessRequest objects
            
        Returns:
            List of successfully processed requests
        """
        processed = []
        
        for request in requests:
            result = self.process_request(request)
            if result is not None:
                processed.append(result)
        
        logger.info(f"Batch processed {len(processed)}/{len(requests)} requests")
        return processed
    
    def _clean_log_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate log data."""
        if data.empty:
            return data
        
        cleaned = data.copy()
        
        # Remove duplicates
        if 'log_id' in cleaned.columns:
            cleaned = cleaned.drop_duplicates(subset=['log_id'])
        else:
            cleaned = cleaned.drop_duplicates()
        
        # Remove invalid timestamps
        if 'timestamp' in cleaned.columns:
            cleaned = cleaned[cleaned['timestamp'].notna()]
        
        # Remove rows with missing critical fields
        critical_fields = ['agent_id', 'action']
        for field in critical_fields:
            if field in cleaned.columns:
                cleaned = cleaned[cleaned[field].notna()]
        
        logger.debug(f"Cleaned data: {len(data)} -> {len(cleaned)} records")
        return cleaned