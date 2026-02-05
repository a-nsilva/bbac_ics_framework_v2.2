#!/usr/bin/env python3
"""
BBAC ICS Framework - Modeling Layer

Implements:
- Baseline Builder (adaptive sliding window)
- Profile Manager (behavioral profiles)
- Feature Engine (feature preparation for analysis)

From flowchart:
- Identify common actions
- Determine normal usage
- Identify normal resources  
- Compute time gaps
- Calculate human presence probability
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..util.config_loader import config
from ..util.data_structures import (
    AgentID,
    AgentType,
    AgentRole,
    ActionType,
    ResourceType,
)


logger = logging.getLogger(__name__)


class BaselineBuilder:
    """
    Builds adaptive baseline from historical data.
    
    Implements sliding window: Baseline(t) = 70% recent + 30% historical
    """
    
    def __init__(self):
        self.baseline_config = config.baseline
        self.window_days = self.baseline_config.get('window_days', 10)
        self.recent_weight = self.baseline_config.get('recent_weight', 0.7)
        self.historical_weight = 1.0 - self.recent_weight
        
        logger.info(
            f"BaselineBuilder: window={self.window_days}d, "
            f"weights=(recent={self.recent_weight}, historical={self.historical_weight})"
        )
    
    def build_baseline(self, historical_data: pd.DataFrame) -> Dict:
        """
        Build baseline from historical data with adaptive weighting.
        
        Args:
            historical_data: DataFrame with baseline_weight column (from sliding window)
            
        Returns:
            Baseline dictionary with weighted statistics
        """
        if historical_data.empty:
            logger.warning("Empty historical data for baseline")
            return self._empty_baseline()
        
        # Check if weights are already applied
        has_weights = 'baseline_weight' in historical_data.columns
        
        baseline = {
            'common_actions': self._identify_common_actions(historical_data, has_weights),
            'normal_usage': self._determine_normal_usage(historical_data, has_weights),
            'normal_resources': self._identify_normal_resources(historical_data, has_weights),
            'time_gaps': self._compute_time_gaps(historical_data, has_weights),
            'human_presence_prob': self._calculate_human_presence(historical_data, has_weights),
            'metadata': {
                'sample_count': len(historical_data),
                'window_days': self.window_days,
                'recent_weight': self.recent_weight,
                'has_adaptive_weights': has_weights,
            }
        }
        
        logger.info(f"Built baseline from {len(historical_data)} samples")
        return baseline
    
    def _identify_common_actions(
        self, 
        data: pd.DataFrame, 
        weighted: bool = False
    ) -> Dict:
        """
        Identify common actions with frequency.
        
        Args:
            data: Historical data
            weighted: Whether to use baseline_weight column
            
        Returns:
            Dictionary with action frequencies
        """
        if 'action' not in data.columns:
            return {}
        
        if weighted and 'baseline_weight' in data.columns:
            # Weighted frequency
            action_weights = data.groupby('action')['baseline_weight'].sum()
            total_weight = data['baseline_weight'].sum()
            action_freq = (action_weights / total_weight).to_dict()
        else:
            # Simple frequency
            action_counts = data['action'].value_counts()
            action_freq = (action_counts / len(data)).to_dict()
        
        # Get top actions (>5% frequency)
        common_actions = {
            action: freq 
            for action, freq in action_freq.items() 
            if freq >= 0.05
        }
        
        return {
            'frequencies': action_freq,
            'common': common_actions,
            'most_common': max(action_freq, key=action_freq.get) if action_freq else None,
            'action_diversity': len(action_freq),
        }
    
    def _determine_normal_usage(
        self, 
        data: pd.DataFrame, 
        weighted: bool = False
    ) -> Dict:
        """
        Determine normal usage patterns.
        
        Args:
            data: Historical data
            weighted: Whether to use baseline_weight column
            
        Returns:
            Dictionary with usage statistics
        """
        usage = {}
        
        # Temporal patterns
        if 'timestamp' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                timestamps = pd.to_datetime(data['timestamp'])
            else:
                timestamps = data['timestamp']
            
            hours = timestamps.dt.hour
            days = timestamps.dt.dayofweek
            
            if weighted and 'baseline_weight' in data.columns:
                # Weighted distribution
                hour_dist = data.groupby(hours)['baseline_weight'].sum()
                hour_dist = (hour_dist / hour_dist.sum()).to_dict()
                
                day_dist = data.groupby(days)['baseline_weight'].sum()
                day_dist = (day_dist / day_dist.sum()).to_dict()
            else:
                # Simple distribution
                hour_dist = hours.value_counts(normalize=True).to_dict()
                day_dist = days.value_counts(normalize=True).to_dict()
            
            usage['hourly_pattern'] = hour_dist
            usage['daily_pattern'] = day_dist
            usage['peak_hours'] = sorted(hour_dist, key=hour_dist.get, reverse=True)[:3]
            usage['active_days'] = sorted(day_dist, key=day_dist.get, reverse=True)[:3]
        
        # Access frequency
        if weighted and 'baseline_weight' in data.columns:
            # Weighted average requests per day
            if 'timestamp' in data.columns:
                data_copy = data.copy()
                data_copy['date'] = pd.to_datetime(data_copy['timestamp']).dt.date
                daily_weights = data_copy.groupby('date')['baseline_weight'].sum()
                usage['avg_requests_per_day'] = float(daily_weights.mean())
        else:
            # Simple count
            if 'timestamp' in data.columns:
                dates = pd.to_datetime(data['timestamp']).dt.date
                usage['avg_requests_per_day'] = float(len(data) / dates.nunique())
        
        # Location patterns
        if 'location' in data.columns:
            if weighted and 'baseline_weight' in data.columns:
                loc_weights = data.groupby('location')['baseline_weight'].sum()
                loc_dist = (loc_weights / loc_weights.sum()).to_dict()
            else:
                loc_dist = data['location'].value_counts(normalize=True).to_dict()
            
            usage['location_distribution'] = loc_dist
            usage['primary_locations'] = sorted(loc_dist, key=loc_dist.get, reverse=True)[:3]
        
        return usage
    
    def _identify_normal_resources(
        self, 
        data: pd.DataFrame, 
        weighted: bool = False
    ) -> Dict:
        """
        Identify normal resources accessed.
        
        Args:
            data: Historical data
            weighted: Whether to use baseline_weight column
            
        Returns:
            Dictionary with resource statistics
        """
        if 'resource' not in data.columns:
            return {}
        
        if weighted and 'baseline_weight' in data.columns:
            # Weighted frequency
            resource_weights = data.groupby('resource')['baseline_weight'].sum()
            total_weight = resource_weights.sum()
            resource_freq = (resource_weights / total_weight).to_dict()
        else:
            # Simple frequency
            resource_freq = data['resource'].value_counts(normalize=True).to_dict()
        
        # Resource types
        resource_types = {}
        if 'resource_type' in data.columns:
            if weighted and 'baseline_weight' in data.columns:
                type_weights = data.groupby('resource_type')['baseline_weight'].sum()
                resource_types = (type_weights / type_weights.sum()).to_dict()
            else:
                resource_types = data['resource_type'].value_counts(normalize=True).to_dict()
        
        return {
            'resource_frequencies': resource_freq,
            'resource_type_distribution': resource_types,
            'unique_resources': int(data['resource'].nunique()),
            'most_accessed': max(resource_freq, key=resource_freq.get) if resource_freq else None,
        }
    
    def _compute_time_gaps(
        self, 
        data: pd.DataFrame, 
        weighted: bool = False
    ) -> Dict:
        """
        Compute time gap statistics.
        
        Args:
            data: Historical data
            weighted: Whether to use baseline_weight column
            
        Returns:
            Dictionary with time gap statistics
        """
        if 'timestamp' not in data.columns or len(data) < 2:
            return {'avg_gap_seconds': 0.0, 'std_gap_seconds': 0.0}
        
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp')
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(data_sorted['timestamp']):
            timestamps = pd.to_datetime(data_sorted['timestamp'])
        else:
            timestamps = data_sorted['timestamp']
        
        # Calculate gaps
        time_diffs = timestamps.diff().dt.total_seconds().dropna()
        
        if len(time_diffs) == 0:
            return {'avg_gap_seconds': 0.0, 'std_gap_seconds': 0.0}
        
        if weighted and 'baseline_weight' in data.columns:
            # Weighted statistics (approximate - pairs get average of their weights)
            weights = data_sorted['baseline_weight'].values
            pair_weights = (weights[:-1] + weights[1:]) / 2
            
            weighted_mean = np.average(time_diffs, weights=pair_weights)
            weighted_var = np.average((time_diffs - weighted_mean)**2, weights=pair_weights)
            weighted_std = np.sqrt(weighted_var)
            
            avg_gap = float(weighted_mean)
            std_gap = float(weighted_std)
        else:
            avg_gap = float(time_diffs.mean())
            std_gap = float(time_diffs.std())
        
        return {
            'avg_gap_seconds': avg_gap,
            'std_gap_seconds': std_gap,
            'min_gap_seconds': float(time_diffs.min()),
            'max_gap_seconds': float(time_diffs.max()),
            'median_gap_seconds': float(time_diffs.median()),
        }
    
    def _calculate_human_presence(
        self, 
        data: pd.DataFrame, 
        weighted: bool = False
    ) -> float:
        """
        Calculate probability of human presence.
        
        Args:
            data: Historical data
            weighted: Whether to use baseline_weight column
            
        Returns:
            Probability [0.0, 1.0]
        """
        if 'human_present' not in data.columns:
            return 0.0
        
        # Convert to boolean if needed
        human_present = data['human_present'].astype(bool)
        
        if weighted and 'baseline_weight' in data.columns:
            # Weighted probability
            weights = data['baseline_weight']
            weighted_sum = (human_present * weights).sum()
            total_weight = weights.sum()
            probability = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            # Simple probability
            probability = human_present.mean()
        
        return float(probability)
    
    def _empty_baseline(self) -> Dict:
        """Return empty baseline structure."""
        return {
            'common_actions': {},
            'normal_usage': {},
            'normal_resources': {},
            'time_gaps': {'avg_gap_seconds': 0.0, 'std_gap_seconds': 0.0},
            'human_presence_prob': 0.0,
            'metadata': {
                'sample_count': 0,
                'window_days': self.window_days,
                'recent_weight': self.recent_weight,
                'has_adaptive_weights': False,
            }
        }


class ProfileManager:
    """
    Manages behavioral profiles for agents.
    
    Stores and retrieves agent-specific baselines and patterns.
    """
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize profile manager.
        
        Args:
            profiles_dir: Directory for storing profiles (from config if None)
        """
        if profiles_dir is None:
            profiles_dir = Path(config.paths.get('profiles_dir', 'profiles'))
        
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.profiles: Dict[AgentID, Dict] = {}
        
        # Separate files for robot/human profiles
        self.robot_profiles_file = self.profiles_dir / 'robot_profiles.json'
        self.human_profiles_file = self.profiles_dir / 'human_profiles.json'
        self.historical_baselines_file = self.profiles_dir / 'historical_baselines.json'
        
        # Load existing profiles
        self._load_profiles()
        
        logger.info(f"ProfileManager initialized: {len(self.profiles)} profiles loaded")
    
    def _load_profiles(self):
        """Load profiles from disk."""
        # Load robot profiles
        if self.robot_profiles_file.exists():
            try:
                with open(self.robot_profiles_file, 'r') as f:
                    robot_profiles = json.load(f)
                self.profiles.update(robot_profiles)
                logger.info(f"Loaded {len(robot_profiles)} robot profiles")
            except Exception as e:
                logger.error(f"Error loading robot profiles: {e}")
        
        # Load human profiles
        if self.human_profiles_file.exists():
            try:
                with open(self.human_profiles_file, 'r') as f:
                    human_profiles = json.load(f)
                self.profiles.update(human_profiles)
                logger.info(f"Loaded {len(human_profiles)} human profiles")
            except Exception as e:
                logger.error(f"Error loading human profiles: {e}")
    
    def save_profiles(self):
        """Save all profiles to disk."""
        # Separate by agent type
        robot_profiles = {}
        human_profiles = {}
        
        for agent_id, profile in self.profiles.items():
            agent_type = profile.get('metadata', {}).get('agent_type')
            
            if agent_type == 'robot' or 'robot' in agent_id.lower():
                robot_profiles[agent_id] = profile
            else:
                human_profiles[agent_id] = profile
        
        # Save robot profiles
        try:
            with open(self.robot_profiles_file, 'w') as f:
                json.dump(robot_profiles, f, indent=2)
            logger.info(f"Saved {len(robot_profiles)} robot profiles")
        except Exception as e:
            logger.error(f"Error saving robot profiles: {e}")
        
        # Save human profiles
        try:
            with open(self.human_profiles_file, 'w') as f:
                json.dump(human_profiles, f, indent=2)
            logger.info(f"Saved {len(human_profiles)} human profiles")
        except Exception as e:
            logger.error(f"Error saving human profiles: {e}")
    
    def create_profile(
        self, 
        agent_id: AgentID,
        baseline: Dict,
        agent_type: AgentType,
        agent_role: AgentRole,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new profile for an agent.
        
        Args:
            agent_id: Agent identifier
            baseline: Baseline dictionary from BaselineBuilder
            agent_type: Robot or human
            agent_role: Specific role
            metadata: Additional metadata
            
        Returns:
            Complete profile dictionary
        """
        import time
        
        profile = {
            'agent_id': agent_id,
            'agent_type': agent_type.value,
            'agent_role': agent_role.value,
            'baseline': baseline,
            'metadata': {
                'created_at': time.time(),
                'updated_at': time.time(),
                'sample_count': baseline.get('metadata', {}).get('sample_count', 0),
                'agent_type': agent_type.value,
                'agent_role': agent_role.value,
            }
        }
        
        # Add custom metadata
        if metadata:
            profile['metadata'].update(metadata)
        
        # Store in cache
        self.profiles[agent_id] = profile
        
        logger.info(f"Created profile for {agent_id} ({agent_type.value}/{agent_role.value})")
        return profile
    
    def update_profile(
        self, 
        agent_id: AgentID,
        new_baseline: Dict,
        trust_score: Optional[float] = None
    ) -> bool:
        """
        Update existing profile with new baseline.
        
        Implements trust filter: only update if trusted.
        
        Args:
            agent_id: Agent identifier
            new_baseline: New baseline to merge
            trust_score: Optional trust score for filtering
            
        Returns:
            True if updated, False otherwise
        """
        import time
        
        # Trust filter (from flowchart continuous learning)
        trust_threshold = config.thresholds.get('trust_threshold', 0.8)
        
        if trust_score is not None and trust_score < trust_threshold:
            logger.warning(
                f"Skipping profile update for {agent_id}: "
                f"trust_score={trust_score:.3f} < threshold={trust_threshold}"
            )
            return False
        
        if agent_id not in self.profiles:
            logger.warning(f"Profile not found for {agent_id}, cannot update")
            return False
        
        # Update baseline
        self.profiles[agent_id]['baseline'] = new_baseline
        self.profiles[agent_id]['metadata']['updated_at'] = time.time()
        self.profiles[agent_id]['metadata']['sample_count'] = (
            new_baseline.get('metadata', {}).get('sample_count', 0)
        )
        
        logger.info(f"Updated profile for {agent_id}")
        return True
    
    def get_profile(self, agent_id: AgentID) -> Optional[Dict]:
        """
        Get profile for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Profile dictionary or None
        """
        return self.profiles.get(agent_id)
    
    def profile_exists(self, agent_id: AgentID) -> bool:
        """Check if profile exists for agent."""
        return agent_id in self.profiles
    
    def list_profiles(self, agent_type: Optional[AgentType] = None) -> List[AgentID]:
        """
        List all profile IDs, optionally filtered by type.
        
        Args:
            agent_type: Optional filter by agent type
            
        Returns:
            List of agent IDs
        """
        if agent_type is None:
            return list(self.profiles.keys())
        
        return [
            agent_id 
            for agent_id, profile in self.profiles.items()
            if profile.get('metadata', {}).get('agent_type') == agent_type.value
        ]
    
    def archive_baseline(self, agent_id: AgentID, baseline: Dict):
        """
        Archive a baseline to historical_baselines.json.
        
        For tracking baseline evolution over time (ADAPTIVE requirement).
        
        Args:
            agent_id: Agent identifier
            baseline: Baseline to archive
        """
        import time
        
        # Load existing archives
        archives = {}
        if self.historical_baselines_file.exists():
            try:
                with open(self.historical_baselines_file, 'r') as f:
                    archives = json.load(f)
            except Exception as e:
                logger.error(f"Error loading historical baselines: {e}")
        
        # Add new entry
        if agent_id not in archives:
            archives[agent_id] = []
        
        archives[agent_id].append({
            'timestamp': time.time(),
            'baseline': baseline,
        })
        
        # Keep only last N archives per agent
        max_archives = config.baseline.get('max_historical_baselines', 10)
        archives[agent_id] = archives[agent_id][-max_archives:]
        
        # Save
        try:
            with open(self.historical_baselines_file, 'w') as f:
                json.dump(archives, f, indent=2)
            logger.debug(f"Archived baseline for {agent_id}")
        except Exception as e:
            logger.error(f"Error archiving baseline: {e}")


class FeatureEngine:
    """
    Prepares features from baseline and current request for analysis layers.
    """
    
    def __init__(self):
        self.baseline_builder = BaselineBuilder()
        
    def extract_request_features(
        self, 
        request_data: Dict,
        profile_baseline: Optional[Dict] = None
    ) -> Dict:
        """
        Extract features from a single request for analysis.
        
        Args:
            request_data: Request data dictionary
            profile_baseline: Optional baseline for comparison
            
        Returns:
            Feature dictionary for analysis layers
        """
        features = {
            # Basic features
            'agent_id': request_data.get('agent_id'),
            'agent_type': request_data.get('agent_type'),
            'agent_role': request_data.get('agent_role'),
            'action': request_data.get('action'),
            'resource': request_data.get('resource'),
            'resource_type': request_data.get('resource_type'),
            'location': request_data.get('location'),
            'human_present': request_data.get('human_present', False),
            'emergency': request_data.get('emergency', False),
            
            # Temporal features
            'hour_of_day': None,
            'day_of_week': None,
            
            # Deviation features (if baseline available)
            'action_is_common': False,
            'resource_is_normal': False,
            'location_is_normal': False,
        }
        
        # Extract temporal features
        timestamp = request_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, (int, float)):
                import datetime
                dt = datetime.datetime.fromtimestamp(timestamp)
                features['hour_of_day'] = dt.hour
                features['day_of_week'] = dt.weekday()
            elif isinstance(timestamp, pd.Timestamp):
                features['hour_of_day'] = timestamp.hour
                features['day_of_week'] = timestamp.dayofweek
        
        # Compare with baseline if available
        if profile_baseline:
            features.update(
                self._compute_deviation_features(request_data, profile_baseline)
            )
        
        return features
    
    def _compute_deviation_features(
        self, 
        request_data: Dict,
        baseline: Dict
    ) -> Dict:
        """
        Compute deviation features by comparing request to baseline.
        
        Args:
            request_data: Current request
            baseline: Agent baseline
            
        Returns:
            Deviation features
        """
        deviations = {}
        
        # Action deviation
        action = request_data.get('action')
        common_actions = baseline.get('common_actions', {}).get('common', {})
        deviations['action_is_common'] = action in common_actions
        
        if common_actions:
            deviations['action_frequency_in_baseline'] = common_actions.get(action, 0.0)
        
        # Resource deviation
        resource = request_data.get('resource')
        resource_freqs = baseline.get('normal_resources', {}).get('resource_frequencies', {})
        deviations['resource_is_normal'] = resource in resource_freqs
        
        if resource_freqs:
            deviations['resource_frequency_in_baseline'] = resource_freqs.get(resource, 0.0)
        
        # Location deviation
        location = request_data.get('location')
        location_dist = baseline.get('normal_usage', {}).get('location_distribution', {})
        deviations['location_is_normal'] = location in location_dist
        
        # Time pattern deviation
        hour = None
        timestamp = request_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, (int, float)):
                import datetime
                hour = datetime.datetime.fromtimestamp(timestamp).hour
            elif isinstance(timestamp, pd.Timestamp):
                hour = timestamp.hour
        
        if hour is not None:
            hourly_pattern = baseline.get('normal_usage', {}).get('hourly_pattern', {})
            if hourly_pattern:
                deviations['hour_frequency_in_baseline'] = hourly_pattern.get(hour, 0.0)
                peak_hours = baseline.get('normal_usage', {}).get('peak_hours', [])
                deviations['is_peak_hour'] = hour in peak_hours
        
        return deviations


class ModelingLayer:
    """Orchestrates the complete modeling pipeline."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize modeling layer.
        
        Args:
            profiles_dir: Optional profiles directory
        """
        self.baseline_builder = BaselineBuilder()
        self.profile_manager = ProfileManager(profiles_dir)
        self.feature_engine = FeatureEngine()
        
        logger.info("ModelingLayer initialized")
    
    def build_agent_profile(
        self,
        agent_id: AgentID,
        historical_data: pd.DataFrame,
        agent_type: AgentType,
        agent_role: AgentRole,
        save: bool = True
    ) -> Dict:
        """
        Build complete profile for an agent.
        
        Args:
            agent_id: Agent identifier
            historical_data: Historical access data
            agent_type: Robot or human
            agent_role: Specific role
            save: Whether to save profile to disk
            
        Returns:
            Profile dictionary
        """
        # Build baseline
        baseline = self.baseline_builder.build_baseline(historical_data)
        
        # Create profile
        profile = self.profile_manager.create_profile(
            agent_id=agent_id,
            baseline=baseline,
            agent_type=agent_type,
            agent_role=agent_role
        )
        
        # Save if requested
        if save:
            self.profile_manager.save_profiles()
        
        logger.info(f"Built profile for {agent_id}")
        return profile
    
    def update_agent_profile(
        self,
        agent_id: AgentID,
        new_historical_data: pd.DataFrame,
        trust_score: Optional[float] = None,
        save: bool = True,
        archive_old: bool = True
    ) -> bool:
        """
        Update agent profile with new data.
        
        Args:
            agent_id: Agent identifier
            new_historical_data: New historical data
            trust_score: Optional trust score for filtering
            save: Whether to save updated profile
            archive_old: Whether to archive old baseline
            
        Returns:
            True if updated successfully
        """
        # Archive current baseline if requested
        if archive_old:
            current_profile = self.profile_manager.get_profile(agent_id)
            if current_profile:
                self.profile_manager.archive_baseline(
                    agent_id, 
                    current_profile['baseline']
                )
        
        # Build new baseline
        new_baseline = self.baseline_builder.build_baseline(new_historical_data)
        
        # Update profile with trust filter
        success = self.profile_manager.update_profile(
            agent_id=agent_id,
            new_baseline=new_baseline,
            trust_score=trust_score
        )
        
        # Save if requested and successful
        if success and save:
            self.profile_manager.save_profiles()
        
        return success
    
    def get_agent_profile(self, agent_id: AgentID) -> Optional[Dict]:
        """Get profile for agent."""
        return self.profile_manager.get_profile(agent_id)
    
    def prepare_features(
        self,
        request_data: Dict,
        agent_id: Optional[AgentID] = None
    ) -> Dict:
        """
        Prepare features for analysis layers.
        
        Args:
            request_data: Request data
            agent_id: Optional agent ID to load profile
            
        Returns:
            Feature dictionary
        """
        # Load profile if agent_id provided
        profile_baseline = None
        if agent_id:
            profile = self.profile_manager.get_profile(agent_id)
            if profile:
                profile_baseline = profile.get('baseline')
        
        # Extract features
        features = self.feature_engine.extract_request_features(
            request_data,
            profile_baseline
        )
        
        return features