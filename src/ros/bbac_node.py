#!/usr/bin/env python3
"""
BBAC ICS Framework - ROS2 Main Node

Main ROS2 node integrating all 5 layers of the BBAC framework.
Provides real-time access control with sub-100ms latency target.

Architecture:
    Layer 1: Ingestion (Auth + Log Collection + Preprocessing)
    Layer 2: Modeling (Baseline + Profiles + Features)
    Layer 3: Analysis (Statistical + Sequence + Policy)
    Layer 4a: Fusion (Score combination)
    Layer 4b: Decision (Risk classification + RBAC)
    Layer 5: Learning (Trust filter + Updates)

ROS2 Topics:
    Subscribers:
        - /access_requests (bbac_msgs/AccessRequest)
    Publishers:
        - /access_decisions (bbac_msgs/AccessDecision)
        - /emergency_alerts (bbac_msgs/EmergencyAlert)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.node import Node

# ROS2 message imports (create these in bbac_msgs package)
from bbac_framework.msg import AccessDecision as ROSAccessDecision
from bbac_framework.msg import AccessRequest as ROSAccessRequest
from bbac_framework.msg import EmergencyAlert
from bbac_framework.msg import LayerDecisionDetail

# BBAC Framework imports
from ..core.ingestion import IngestionLayer
from ..core.modeling import ModelingLayer
from ..core.analysis import AnalysisLayer
from ..core.fusion import FusionLayer
from ..core.decision import DecisionLayer
from ..core.learning import ContinuousLearningLayer
from data.loader import DatasetLoader
from ..util.config_loader import config
from ..util.data_structures import (
    AccessRequest,
    AccessDecision,
    AgentType,
    AgentRole,
    ActionType,
    AuthStatus,
    ResourceType,
    DecisionType,
)


logger = logging.getLogger(__name__)


class BBACNode(Node):
    """
    BBAC Hybrid Access Control Engine - ROS2 Node.
    
    Integrates all 5 layers and provides real-time access control
    with adaptive learning and dynamic rule updates.
    """
    
    def __init__(self) -> None:
        """Initialize BBAC Engine node."""
        super().__init__('bbac_node')
        
        # Declare ROS parameters
        self._declare_parameters()
        
        # Get parameter values
        self.enable_behavioral = self.get_parameter('enable_behavioral').value
        self.enable_ml = self.get_parameter('enable_ml').value
        self.enable_policy = self.get_parameter('enable_policy').value
        self.data_path = Path(self.get_parameter('data_path').value)
        self.profiles_path = Path(self.get_parameter('profiles_path').value)
        self.log_path = Path(self.get_parameter('log_path').value)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("BBAC FRAMEWORK - ROS2 NODE INITIALIZATION")
        self.get_logger().info("=" * 70)
        self.get_logger().info(
            f"Configuration: behavioral={self.enable_behavioral}, "
            f"ml={self.enable_ml}, policy={self.enable_policy}"
        )
        
        # Initialize dataset loader
        self._initialize_dataset()
        
        # Initialize all 5 layers
        self._initialize_layers()
        
        # Setup ROS2 communication
        self._setup_ros_communication()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'grants': 0,
            'denials': 0,
            'approvals': 0,
            'avg_latency_ms': 0.0,
        }
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("BBAC NODE READY")
        self.get_logger().info(f"Subscribed to: /access_requests")
        self.get_logger().info(f"Publishing to: /access_decisions, /emergency_alerts")
        self.get_logger().info("=" * 70)
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        self.declare_parameter('enable_behavioral', True)
        self.declare_parameter('enable_ml', True)
        self.declare_parameter('enable_policy', True)
        self.declare_parameter('data_path', 'data/100k')
        self.declare_parameter('profiles_path', 'profiles')
        self.declare_parameter('log_path', 'logs')
        self.declare_parameter('target_latency_ms', 100.0)
    
    def _initialize_dataset(self):
        """Initialize dataset loader."""
        try:
            self.dataset_loader = DatasetLoader(self.data_path)
            self.dataset_loader.load_all()
            self.get_logger().info("✓ Dataset loaded successfully")
        except Exception as e:
            self.get_logger().warning(f"Dataset not loaded: {e}")
            self.dataset_loader = None
    
    def _initialize_layers(self):
        """Initialize all 5 BBAC layers."""
        start_time = time.time()
        
        # Layer 1: Ingestion
        self.get_logger().info("Initializing Layer 1: Ingestion...")
        self.ingestion_layer = IngestionLayer(self.dataset_loader)
        self.get_logger().info("✓ Layer 1 (Ingestion) ready")
        
        # Layer 2: Modeling
        self.get_logger().info("Initializing Layer 2: Modeling...")
        self.modeling_layer = ModelingLayer(self.profiles_path)
        self.get_logger().info(
            f"✓ Layer 2 (Modeling) ready - "
            f"{len(self.modeling_layer.profile_manager.profiles)} profiles loaded"
        )
        
        # Layer 3: Analysis
        self.get_logger().info("Initializing Layer 3: Analysis...")
        self.analysis_layer = AnalysisLayer()
        
        # Build sequence models from profiles
        if self.dataset_loader:
            self._build_sequence_models()
        
        self.get_logger().info("✓ Layer 3 (Analysis) ready")
        
        # Layer 4a: Fusion
        self.get_logger().info("Initializing Layer 4a: Fusion...")
        self.fusion_layer = FusionLayer(use_meta_classifier=False)
        self.get_logger().info("✓ Layer 4a (Fusion) ready")
        
        # Layer 4b: Decision
        self.get_logger().info("Initializing Layer 4b: Decision...")
        self.decision_layer = DecisionLayer(self.log_path)
        self.get_logger().info("✓ Layer 4b (Decision) ready")
        
        # Layer 5: Learning
        self.get_logger().info("Initializing Layer 5: Learning...")
        self.learning_layer = ContinuousLearningLayer(
            profile_manager=self.modeling_layer.profile_manager,
            log_dir=self.log_path
        )
        self.get_logger().info("✓ Layer 5 (Learning) ready")
        
        init_time = (time.time() - start_time) * 1000
        self.get_logger().info(f"All 5 layers initialized in {init_time:.2f}ms")
    
    def _build_sequence_models(self):
        """Build Markov chain models for sequence analysis."""
        try:
            # Get training data
            train_data = self.dataset_loader.train_data
            
            if train_data is None or train_data.empty:
                self.get_logger().warning("No training data for sequence models")
                return
            
            # Group by agent
            agents = train_data['agent_id'].unique()
            
            for agent_id in agents:
                agent_data = train_data[train_data['agent_id'] == agent_id]
                
                # Build transition matrix
                self.analysis_layer.sequence_engine.build_transition_matrix(
                    agent_id,
                    agent_data
                )
            
            self.get_logger().info(f"Built sequence models for {len(agents)} agents")
            
        except Exception as e:
            self.get_logger().error(f"Error building sequence models: {e}")
    
    def _setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers."""
        # Subscriber for access requests
        self.request_sub = self.create_subscription(
            ROSAccessRequest,
            '/access_requests',
            self._handle_request_callback,
            10
        )
        
        # Publisher for access decisions
        self.decision_pub = self.create_publisher(
            ROSAccessDecision,
            '/access_decisions',
            10
        )
        
        # Publisher for emergency alerts
        self.alert_pub = self.create_publisher(
            EmergencyAlert,
            '/emergency_alerts',
            10
        )
        
        self.get_logger().info("ROS2 communication setup complete")
    
    def _handle_request_callback(self, ros_msg: ROSAccessRequest):
        """
        Callback for incoming access requests.
        
        Args:
            ros_msg: ROS AccessRequest message
        """
        start_time = time.time()
        request_id = ros_msg.request_id
        
        try:
            self.get_logger().debug(f"Received request: {request_id}")
            
            # Convert ROS message to dataclass
            request = self._ros_to_dataclass(ros_msg)
            
            # Process through all 5 layers
            decision = self._process_request(request)
            
            # Convert dataclass to ROS message and publish
            ros_decision = self._dataclass_to_ros(decision, ros_msg.header)
            self.decision_pub.publish(ros_decision)
            
            # Update statistics
            self._update_statistics(decision, time.time() - start_time)
            
            # Log
            self.get_logger().info(
                f"Request {request_id}: {decision.decision.value} "
                f"(conf={decision.confidence:.3f}, latency={decision.latency_ms:.2f}ms)"
            )
            
            # Publish alert if high-confidence denial
            if (decision.decision == DecisionType.DENY and 
                decision.confidence > 0.8):
                self._publish_alert(request, decision)
            
        except Exception as e:
            self.get_logger().error(f"Error processing request {request_id}: {e}")
            # Publish denial on error
            self._publish_error_decision(ros_msg, str(e))
    
    def _ros_to_dataclass(self, ros_msg: ROSAccessRequest) -> AccessRequest:
        """
        Convert ROS message to AccessRequest dataclass.
        
        Args:
            ros_msg: ROS AccessRequest message
            
        Returns:
            AccessRequest dataclass
        """
        # Parse enums
        agent_type = AgentType.from_string(ros_msg.agent_type)
        agent_role = AgentRole.from_string(ros_msg.agent_role, agent_type)
        action = ActionType.from_string(ros_msg.action)
        resource_type = ResourceType.from_string(ros_msg.resource_type)
        auth_status = AuthStatus.from_string(ros_msg.auth_status)
        
        # Parse previous action
        previous_action = None
        if ros_msg.previous_action:
            try:
                previous_action = ActionType.from_string(ros_msg.previous_action)
            except ValueError:
                pass
        
        # Parse context_data
        context = {}
        for item in ros_msg.context_data:
            if ':' in item:
                key, value = item.split(':', 1)
                context[key] = value
        
        # Create AccessRequest
        return AccessRequest(
            request_id=ros_msg.request_id,
            timestamp=ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9,
            agent_id=ros_msg.agent_id,
            agent_type=agent_type,
            agent_role=agent_role,
            action=action,
            resource=ros_msg.resource,
            resource_type=resource_type,
            location=ros_msg.location,
            human_present=ros_msg.human_present,
            emergency=ros_msg.emergency,
            session_id=ros_msg.session_id if ros_msg.session_id else None,
            previous_action=previous_action,
            auth_status=auth_status,
            attempt_count=int(ros_msg.attempt_count),
            policy_id=ros_msg.policy_id if ros_msg.policy_id else None,
            zone=ros_msg.zone if ros_msg.zone else None,
            priority=float(ros_msg.priority),
            context=context,
        )
    
    def _dataclass_to_ros(
        self,
        decision: AccessDecision,
        header
    ) -> ROSAccessDecision:
        """
        Convert AccessDecision dataclass to ROS message.
        
        Args:
            decision: AccessDecision dataclass
            header: Original request header
            
        Returns:
            ROS AccessDecision message
        """
        ros_decision = ROSAccessDecision()
        ros_decision.header = header
        ros_decision.request_id = decision.request_id
        ros_decision.decision = decision.decision.value
        ros_decision.confidence = float(decision.confidence)
        ros_decision.latency_ms = float(decision.latency_ms)
        ros_decision.reason = decision.reason
        
        # Convert layer decisions
        ros_decision.layer_decisions = []
        for layer_name, layer_info in decision.layer_decisions.items():
            detail = LayerDecisionDetail()
            detail.layer_name = layer_name
            
            # Handle different layer_info formats
            if isinstance(layer_info, dict):
                detail.decision = str(layer_info.get('decision', 'unknown'))
                detail.confidence = float(layer_info.get('confidence', 0.0))
                detail.latency_ms = float(layer_info.get('latency_ms', 0.0))
                
                # Convert explanation to key-value pairs
                explanation = layer_info.get('explanation', {})
                if isinstance(explanation, dict):
                    for key, value in explanation.items():
                        detail.explanation_keys.append(str(key))
                        detail.explanation_values.append(json.dumps(value))
            
            ros_decision.layer_decisions.append(detail)
        
        ros_decision.logged = True
        ros_decision.log_id = f"log_{decision.request_id}"
        
        return ros_decision
    
    def _process_request(self, request: AccessRequest) -> AccessDecision:
        """
        Process access request through all 5 layers.
        
        Pipeline:
            1. Ingestion: Validate and preprocess
            2. Modeling: Extract features
            3. Analysis: Stat + Sequence + Policy
            4a. Fusion: Combine scores
            4b. Decision: Risk classification + RBAC
            5. Learning: Trust filter + updates
        
        Args:
            request: AccessRequest dataclass
            
        Returns:
            AccessDecision dataclass
        """
        # Layer 1: Ingestion (authentication + validation)
        processed_request = self.ingestion_layer.process_request(request)
        
        if processed_request is None:
            # Authentication failed
            return self._create_auth_failed_decision(request)
        
        # Layer 2: Modeling (feature extraction)
        profile = self.modeling_layer.get_agent_profile(request.agent_id)
        
        profile_baseline = profile.get('baseline') if profile else None
        
        features = self.modeling_layer.prepare_features(
            request_data={
                'agent_id': request.agent_id,
                'agent_type': request.agent_type.value,
                'agent_role': request.agent_role.value,
                'action': request.action.value,
                'resource': request.resource,
                'resource_type': request.resource_type.value,
                'location': request.location,
                'human_present': request.human_present,
                'emergency': request.emergency,
                'timestamp': request.timestamp,
            },
            agent_id=request.agent_id
        )
        
        # Layer 3: Analysis (statistical + sequence + policy)
        layer_results = self.analysis_layer.analyze_request(
            request=processed_request,
            profile_baseline=profile_baseline,
            features=features,
            enable_stat=self.enable_behavioral,
            enable_sequence=self.enable_ml,
            enable_policy=self.enable_policy,
        )
        
        # Layer 4a: Fusion
        fused_result = self.fusion_layer.fuse(layer_results)
        
        # Layer 4b: Decision
        decision = self.decision_layer.make_decision(
            request=processed_request,
            fused_result=fused_result
        )
        
        # Layer 5: Learning (post-decision)
        fused_score = fused_result['confidence']
        self.learning_layer.process_decision(
            request=processed_request,
            decision=decision,
            fused_score=fused_score
        )
        
        return decision
    
    def _create_auth_failed_decision(self, request: AccessRequest) -> AccessDecision:
        """Create decision for authentication failure."""
        return AccessDecision(
            request_id=request.request_id,
            timestamp=time.time(),
            decision=DecisionType.DENY,
            confidence=1.0,
            latency_ms=1.0,
            reason="Authentication failed",
            layer_decisions={
                'authentication': {
                    'decision': 'deny',
                    'confidence': 1.0,
                    'reason': 'auth_failed',
                }
            }
        )
    
    def _publish_error_decision(self, ros_msg: ROSAccessRequest, error: str):
        """Publish denial decision on error."""
        ros_decision = ROSAccessDecision()
        ros_decision.header = ros_msg.header
        ros_decision.request_id = ros_msg.request_id
        ros_decision.decision = 'deny'
        ros_decision.confidence = 1.0
        ros_decision.latency_ms = 1.0
        ros_decision.reason = f"Processing error: {error}"
        ros_decision.logged = False
        
        self.decision_pub.publish(ros_decision)
    
    def _publish_alert(self, request: AccessRequest, decision: AccessDecision):
        """Publish emergency alert for high-confidence denial."""
        alert = EmergencyAlert()
        alert.header.stamp = self.get_clock().now().to_msg()
        alert.emergency_type = 'access_denied'
        alert.severity = 'high'
        alert.priority = 80
        alert.active = True
        alert.status = 'triggered'
        alert.zone = request.location
        alert.affected_resources = [request.resource]
        alert.required_actions = ['log_incident', 'notify_supervisor']
        alert.notifications = ['supervisor', 'security_officer']
        alert.description = (
            f"High-confidence access denial for agent {request.agent_id} "
            f"attempting {request.action.value} on {request.resource}"
        )
        
        self.alert_pub.publish(alert)
        self.get_logger().warn(f"Alert published for request {request.request_id}")
    
    def _update_statistics(self, decision: AccessDecision, processing_time: float):
        """Update node statistics."""
        self.stats['total_requests'] += 1
        
        if decision.decision == DecisionType.GRANT:
            self.stats['grants'] += 1
        elif decision.decision == DecisionType.DENY:
            self.stats['denials'] += 1
        elif decision.decision == DecisionType.REQUIRE_APPROVAL:
            self.stats['approvals'] += 1
        
        # Update average latency
        total = self.stats['total_requests']
        current_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (
            (current_avg * (total - 1) + decision.latency_ms) / total
        )
    
    def get_statistics(self) -> Dict:
        """Get node statistics."""
        stats = self.stats.copy()
        
        # Add layer statistics
        stats['decision_layer'] = self.decision_layer.get_statistics()
        stats['learning_layer'] = self.learning_layer.get_statistics()
        
        return stats
    
    def trigger_emergency(self, emergency_type: str) -> bool:
        """
        Trigger emergency state.
        
        Args:
            emergency_type: Type of emergency
            
        Returns:
            True if triggered successfully
        """
        try:
            alert = EmergencyAlert()
            alert.header.stamp = self.get_clock().now().to_msg()
            alert.emergency_type = emergency_type
            alert.severity = 'critical'
            alert.priority = 100
            alert.active = True
            alert.status = 'triggered'
            alert.description = f"Emergency triggered: {emergency_type}"
            
            self.alert_pub.publish(alert)
            self.get_logger().warn(f"Emergency triggered: {emergency_type}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error triggering emergency: {e}")
            return False


def main(args=None):
    """Main entry point for BBAC ROS2 node."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node
    node = BBACNode()
    
    try:
        # Spin node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested...")
        
    finally:
        # Cleanup
        node.get_logger().info("Saving profiles and statistics...")
        
        # Save profiles
        node.modeling_layer.profile_manager.save_profiles()
        
        # Save anomaly patterns
        node.learning_layer.anomaly_logger.save_patterns()
        
        # Print final statistics
        stats = node.get_statistics()
        node.get_logger().info("=" * 70)
        node.get_logger().info("FINAL STATISTICS")
        node.get_logger().info(f"Total requests: {stats['total_requests']}")
        node.get_logger().info(f"Grants: {stats['grants']}")
        node.get_logger().info(f"Denials: {stats['denials']}")
        node.get_logger().info(f"Approvals: {stats['approvals']}")
        node.get_logger().info(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
        node.get_logger().info("=" * 70)
        
        # Shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()