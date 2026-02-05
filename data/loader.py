#!/usr/bin/env python3
"""
BBAC ICS Framework - Dataset Loader
Loads real dataset and converts to framework structures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.util.config_loader import config
from src.util.data_structures import (
    AccessRequest,
    AgentType,
    AgentRole,
    ActionType,
    AuthStatus,
    ResourceType,
)


logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and manages the BBAC dataset with structure conversion."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Optional custom path, uses config by default
        """
        self.paths_config = config.paths
        
        # Use provided path or default from config
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            self.dataset_path = Path(
                self.paths_config.get('data_dir', 'data/100k')
            )
        
        # Validate dataset exists
        self._validate_dataset()
        
        # Initialize data containers
        self.train_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.agents_metadata: Optional[Dict] = None
        self.anomaly_metadata: Optional[Dict] = None
        
        logger.info(f"DatasetLoader initialized with path: {self.dataset_path}")
    
    def _validate_dataset(self):
        """Validate that required dataset files exist."""
        # APENAS os CSVs são obrigatórios
        required_files = [
            self.paths_config.get('train_file', 'bbac_trainer.csv'),
            self.paths_config.get('validation_file', 'bbac_validation.csv'),
            self.paths_config.get('test_file', 'bbac_test.csv'),
        ]
        
        missing_files = []
        for filename in required_files:
            filepath = self.dataset_path / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            error_msg = (
                f"\n{'='*50}\n"
                f"Dataset incomplete at: {self.dataset_path}\n"
                f"{'='*50}\n"
                f"Missing required files:\n"
            )
            for filename in missing_files:
                error_msg += f"  - {filename}\n"
            error_msg += f"{'='*50}"
            raise FileNotFoundError(error_msg)
        
        # Opcional: avisar sobre arquivos opcionais ausentes
        optional_files = ['agents.json', 'anomaly_metadata.json']
        missing_optional = []
        
        for filename in optional_files:
            filepath = self.dataset_path / filename
            if not filepath.exists():
                missing_optional.append(filename)
        
        if missing_optional:
            logger.info(
                f"Optional files not found (OK to skip): {', '.join(missing_optional)}"
            )
    
    def load_all(self) -> bool:
        """Load all dataset files."""
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            
            # Carregar CSVs obrigatórios
            self.train_data = self._load_csv_file('train_file')
            self.validation_data = self._load_csv_file('validation_file')
            self.test_data = self._load_csv_file('test_file')
            
            # Carregar JSONs opcionais
            self.agents_metadata = self._load_optional_json('agents.json')
            self.anomaly_metadata = self._load_optional_json('anomaly_metadata.json')
            
            logger.info("✓ All required dataset files loaded successfully")
            self._print_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error loading dataset: {e}")
            return False
    
    def _load_csv_file(self, config_key: str) -> pd.DataFrame:
        """Load CSV file with proper parsing."""
        filename = self.paths_config.get(config_key)
        if not filename:
            raise ValueError(f"No filename configured for {config_key}")
        
        filepath = self.dataset_path / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        
        # Parse timestamps if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(
                df['timestamp'],
                format='mixed',
                errors='coerce'
            )
        
        # Standardize column names
        column_mapping = {
            'user_id': 'agent_id',
            'resource_id': 'resource',
        }
        
        df = df.rename(columns={
            old: new for old, new in column_mapping.items()
            if old in df.columns
        })
        
        logger.debug(f"Loaded {len(df)} samples from {filename}")
        return df
    
    def _load_optional_json(self, filename: str) -> Optional[Dict]:
        """
        Load optional JSON file.
        
        Args:
            filename: Name of JSON file
            
        Returns:
            Dictionary with JSON content, or None if file doesn't exist
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            logger.debug(f"Optional JSON file not found: {filename} (skipping)")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded optional file: {filename}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {filename}: {e} (skipping)")
            return None
        except Exception as e:
            logger.warning(f"Error loading {filename}: {e} (skipping)")
            return None
    
    def to_access_request(self, row: pd.Series) -> AccessRequest:
        """
        Convert dataset row to AccessRequest structure.
        
        Maps 17 dataset fields:
        - log_id → request_id
        - timestamp → timestamp (parsed to float)
        - session_id → session_id
        - user_id → agent_id
        - agent_type → agent_type (enum)
        - robot_type/human_role → agent_role (consolidated enum)
        - action → action (enum)
        - resource → resource (string)
        - resource_type → resource_type (enum)
        - human_present → human_present (bool)
        - emergency_flag → emergency (bool)
        - location → location (string)
        - previous_action → previous_action (enum)
        - auth_status → auth_status (enum)
        - attempt_count → attempt_count (int)
        - policy_id → policy_id (string)
        
        Args:
            row: DataFrame row from dataset
            
        Returns:
            AccessRequest structure
        """
        # Core identifiers
        request_id = str(row.get('log_id', ''))
        
        # Parse timestamp
        timestamp_raw = row.get('timestamp')
        if pd.isna(timestamp_raw):
            timestamp = 0.0
        elif isinstance(timestamp_raw, pd.Timestamp):
            timestamp = timestamp_raw.timestamp()
        else:
            timestamp = float(timestamp_raw)
        
        agent_id = str(row.get('agent_id', row.get('user_id', '')))
        
        # Parse agent type
        agent_type_str = str(row.get('agent_type', 'robot')).strip()
        try:
            agent_type = AgentType.from_string(agent_type_str)
        except ValueError:
            logger.warning(f"Unknown agent_type '{agent_type_str}', defaulting to ROBOT")
            agent_type = AgentType.ROBOT
        
        # Parse agent role (consolidates robot_type + human_role)
        robot_type = row.get('robot_type', '')
        human_role = row.get('human_role', '')
        
        # Prioritize non-empty field
        role_str = robot_type if (robot_type and str(robot_type).strip()) else human_role
        
        try:
            agent_role = AgentRole.from_string(str(role_str), agent_type)
        except (ValueError, AttributeError):
            logger.debug(f"Could not parse role '{role_str}', defaulting to UNKNOWN")
            agent_role = AgentRole.UNKNOWN
        
        # Parse action
        action_str = str(row.get('action', 'read')).strip()
        try:
            action = ActionType.from_string(action_str)
        except ValueError:
            logger.warning(f"Unknown action '{action_str}', defaulting to READ")
            action = ActionType.READ
        
        # Resource fields
        resource = str(row.get('resource', ''))
        
        resource_type_str = str(row.get('resource_type', 'other')).strip()
        try:
            resource_type = ResourceType.from_string(resource_type_str)
        except ValueError:
            resource_type = ResourceType.OTHER
        
        # Context fields
        location = str(row.get('location', 'unknown'))
        
        human_present_raw = row.get('human_present', False)
        if pd.isna(human_present_raw):
            human_present = False
        elif isinstance(human_present_raw, bool):
            human_present = human_present_raw
        elif isinstance(human_present_raw, str):
            human_present = human_present_raw.lower() in ('true', '1', 'yes')
        else:
            human_present = bool(human_present_raw)
        
        emergency_raw = row.get('emergency_flag', False)
        if pd.isna(emergency_raw):
            emergency = False
        elif isinstance(emergency_raw, bool):
            emergency = emergency_raw
        elif isinstance(emergency_raw, str):
            emergency = emergency_raw.lower() in ('true', '1', 'yes')
        else:
            emergency = bool(emergency_raw)
        
        # Session tracking
        session_id_raw = row.get('session_id')
        session_id = str(session_id_raw) if not pd.isna(session_id_raw) else None
        
        previous_action_str = row.get('previous_action')
        previous_action = None
        if not pd.isna(previous_action_str) and str(previous_action_str).strip():
            try:
                previous_action = ActionType.from_string(str(previous_action_str))
            except ValueError:
                pass
        
        # Authentication fields
        auth_status_str = row.get('auth_status', 'success')
        if pd.isna(auth_status_str):
            auth_status = AuthStatus.SUCCESS
        else:
            try:
                auth_status = AuthStatus.from_string(str(auth_status_str))
            except ValueError:
                auth_status = AuthStatus.SUCCESS
        
        attempt_count_raw = row.get('attempt_count', 0)
        attempt_count = int(attempt_count_raw) if not pd.isna(attempt_count_raw) else 0
        
        # Policy tracking
        policy_id_raw = row.get('policy_id')
        policy_id = str(policy_id_raw) if not pd.isna(policy_id_raw) else None
        
        # Create AccessRequest
        return AccessRequest(
            request_id=request_id,
            timestamp=timestamp,
            agent_id=agent_id,
            agent_type=agent_type,
            agent_role=agent_role,
            action=action,
            resource=resource,
            resource_type=resource_type,
            location=location,
            human_present=human_present,
            emergency=emergency,
            session_id=session_id,
            previous_action=previous_action,
            auth_status=auth_status,
            attempt_count=attempt_count,
            policy_id=policy_id,
        )
    
    def get_requests_from_dataframe(
        self, 
        df: pd.DataFrame,
        max_requests: Optional[int] = None
    ) -> List[AccessRequest]:
        """
        Convert DataFrame to list of AccessRequest structures.
        
        Args:
            df: DataFrame to convert
            max_requests: Optional limit on number of requests
            
        Returns:
            List of AccessRequest objects
        """
        if df is None or df.empty:
            return []
        
        # Limit if specified
        if max_requests:
            df = df.head(max_requests)
        
        requests = []
        for idx, row in df.iterrows():
            try:
                request = self.to_access_request(row)
                requests.append(request)
            except Exception as e:
                logger.error(f"Error converting row {idx}: {e}")
                continue
        
        logger.info(f"Converted {len(requests)} rows to AccessRequest structures")
        return requests
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if self.train_data is None:
            return
        
        stats = self.get_statistics()
        
        logger.info("=" * 50)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Training samples:    {stats['train_samples']:,}")
        logger.info(f"Validation samples:  {stats['validation_samples']:,}")
        logger.info(f"Test samples:        {stats['test_samples']:,}")
        
        if 'unique_agents' in stats:
            logger.info(f"Unique agents:       {stats['unique_agents']:,}")
        
        if 'unique_resources' in stats:
            logger.info(f"Unique resources:    {stats['unique_resources']:,}")
        
        if 'unique_actions' in stats:
            logger.info(f"Unique actions:      {stats['unique_actions']:,}")
        
        if 'anomaly_count' in stats:
            logger.info(f"Anomaly samples:     {stats['anomaly_count']:,}")
        
        logger.info("=" * 50)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {
            'train_samples': len(self.train_data) if self.train_data is not None else 0,
            'validation_samples': len(self.validation_data) if self.validation_data is not None else 0,
            'test_samples': len(self.test_data) if self.test_data is not None else 0,
            'dataset_path': str(self.dataset_path),
        }
        
        # Add metadata counts (opcional)
        if self.agents_metadata:
            stats['agent_profiles'] = len(self.agents_metadata)
        
        if self.anomaly_metadata:
            stats['anomaly_types'] = len(self.anomaly_metadata.get('anomaly_types', []))
        
        # Add column-based statistics
        if self.train_data is not None:
            for column in ['agent_id', 'resource', 'action']:
                if column in self.train_data.columns:
                    stats[f'unique_{column}s'] = self.train_data[column].nunique()
            
            # Count anomalies if column exists
            if 'is_anomaly' in self.train_data.columns:
                stats['anomaly_count'] = self.train_data['is_anomaly'].sum()
                stats['anomaly_percentage'] = (
                    stats['anomaly_count'] / len(self.train_data) * 100
                )
        
        return stats
    
    def get_training_data(self, include_validation: bool = False) -> pd.DataFrame:
        """
        Get training data, optionally including validation data.
        
        Args:
            include_validation: Whether to include validation data
            
        Returns:
            Combined training data
        """
        if self.train_data is None:
            raise ValueError("Training data not loaded")
        
        if include_validation and self.validation_data is not None:
            combined = pd.concat([self.train_data, self.validation_data], ignore_index=True)
            logger.info(f"Combined training+validation: {len(combined):,} samples")
            return combined
        
        return self.train_data
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific agent (optional).
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent metadata dictionary, or None if not available
        """
        if not self.agents_metadata:
            logger.debug(f"No agents metadata available for {agent_id}")
            return None
        
        return self.agents_metadata.get(agent_id)
    
    def get_anomaly_types(self) -> List[str]:
        """
        Get list of anomaly types in the dataset (optional).
        
        Returns:
            List of anomaly type names, or empty list if not available
        """
        if not self.anomaly_metadata:
            logger.debug("No anomaly metadata available")
            return []
        
        return self.anomaly_metadata.get('anomaly_types', [])
    
    def get_data_split(self, split: str = 'train') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get features and labels for a specific split.
        
        Args:
            split: 'train', 'validation', or 'test'
            
        Returns:
            Tuple of (features, labels)
            labels may be None if no label columns present
        """
        if split == 'train':
            data = self.train_data
        elif split == 'validation':
            data = self.validation_data
        elif split == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if data is None:
            raise ValueError(f"{split} data not loaded")
        
        # Features: all columns except label columns
        label_columns = ['is_anomaly', 'anomaly_type', 'anomaly_severity', 'expected_decision']
        feature_columns = [col for col in data.columns if col not in label_columns]
        
        features = data[feature_columns].copy()
        
        # Labels: only if at least one label column exists
        label_cols_present = [col for col in label_columns if col in data.columns]
        if label_cols_present:
            labels = data[label_cols_present].copy()
        else:
            labels = None
        
        return features, labels