#!/usr/bin/env python3
"""
Basic smoke tests for BBAC framework.
"""

#import sys
#from pathlib import Path

# Add src to path
#sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from src.util.data_structures import AccessRequest, AgentType, ActionType, ResourceType


def test_import_layers():
    """Test that all layers can be imported."""
    from src.core.ingestion import IngestionLayer
    from src.core.modeling import ModelingLayer
    from src.core.analysis import AnalysisLayer
    from src.core.fusion import FusionLayer
    from src.core.decision import DecisionLayer
    from src.core.learning import ContinuousLearningLayer
    
    assert IngestionLayer is not None
    assert ModelingLayer is not None
    assert AnalysisLayer is not None
    assert FusionLayer is not None
    assert DecisionLayer is not None
    assert ContinuousLearningLayer is not None


def test_import_models():
    """Test that all models can be imported."""
    from src.models.statistical import StatisticalModel, IsolationForestModel
    from src.models.fusion import MetaClassifier, EnsembleFusion
    from src.models.lstm import LSTMSequenceModel
    
    assert StatisticalModel is not None
    assert IsolationForestModel is not None
    assert MetaClassifier is not None
    assert EnsembleFusion is not None
    assert LSTMSequenceModel is not None


def test_access_request_creation():
    """Test AccessRequest creation."""
    request = AccessRequest(
        request_id='test_001',
        agent_id='robot_01',
        agent_type=AgentType.ROBOT,
        agent_role='assembly_robot',
        action=ActionType.READ,
        resource='sensor_01',
        resource_type=ResourceType.SENSOR,
        location='assembly_line',
        human_present=False,
        emergency=False,
    )
    
    assert request.request_id == 'test_001'
    assert request.agent_type == AgentType.ROBOT
    assert request.action == ActionType.READ


def test_config_loader():
    """Test config loader."""
    from src.util.config_loader import config
    
    assert config is not None
    assert hasattr(config, 'baseline')
    assert hasattr(config, 'fusion')
    assert hasattr(config, 'thresholds')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])