#!/usr/bin/env python3
"""
BBAC ICS Framework - Robot Agent Simulator

Simulates robot agents publishing access requests to BBAC node.
Generates predictable behavioral patterns for different robot types.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node

from bbac_framework.msg import AccessDecision as ROSAccessDecision
from bbac_framework.msg import AccessRequest as ROSAccessRequest

from ..util.data_structures import AgentRole, ActionType, ResourceType


logger = logging.getLogger(__name__)


class RobotBehaviorProfile:
    """Defines behavioral pattern for a robot type."""
    
    def __init__(
        self,
        robot_type: str,
        common_actions: List[str],
        common_resources: List[str],
        resource_types: List[str],
        locations: List[str],
        request_rate: float = 1.0,  # requests per second
        pattern_variance: float = 0.1,  # 10% variance
    ):
        """
        Initialize robot behavior profile.
        
        Args:
            robot_type: Robot type (assembly, transport, camera, inspection)
            common_actions: List of common actions
            common_resources: List of common resources
            resource_types: List of resource types
            locations: Typical locations
            request_rate: Requests per second
            pattern_variance: Variance in behavior (0.0 = deterministic, 1.0 = random)
        """
        self.robot_type = robot_type
        self.common_actions = common_actions
        self.common_resources = common_resources
        self.resource_types = resource_types
        self.locations = locations
        self.request_rate = request_rate
        self.pattern_variance = pattern_variance
        
        # Markov-like action sequence
        self.action_sequences = {
            'read': ['execute', 'read', 'write'],
            'execute': ['read', 'execute'],
            'write': ['read', 'execute'],
        }
        
        self.last_action = 'read'
    
    def generate_request(self, agent_id: str, request_id: str) -> Dict:
        """Generate access request based on behavior profile."""
        # Action selection (weighted by pattern)
        if random.random() < self.pattern_variance:
            # Random action (variance)
            action = random.choice(['read', 'write', 'execute', 'delete'])
        else:
            # Predictable action (follow sequence)
            possible_actions = self.action_sequences.get(
                self.last_action,
                self.common_actions
            )
            action = random.choice(possible_actions)
        
        self.last_action = action
        
        # Resource selection (weighted)
        if random.random() < self.pattern_variance:
            resource_type = random.choice(list(ResourceType))
            resource = f"{resource_type.value}_unknown_{random.randint(1, 100)}"
        else:
            resource = random.choice(self.common_resources)
            # Infer resource type from resource name
            for rt in self.resource_types:
                if rt.lower() in resource.lower():
                    resource_type = rt
                    break
            else:
                resource_type = random.choice(self.resource_types)
        
        # Location
        location = random.choice(self.locations)
        
        # Context
        human_present = random.random() < 0.3  # 30% chance human present
        emergency = random.random() < 0.01  # 1% chance emergency
        
        return {
            'request_id': request_id,
            'agent_id': agent_id,
            'agent_type': 'robot',
            'agent_role': self.robot_type,
            'action': action,
            'resource': resource,
            'resource_type': resource_type,
            'location': location,
            'human_present': human_present,
            'emergency': emergency,
            'session_id': f"session_{agent_id}_{int(time.time())}",
            'previous_action': self.last_action,
            'auth_status': 'success',
            'attempt_count': 0,
            'policy_id': '',
            'zone': location,
            'priority': 5.0,
        }


# Predefined robot profiles
ROBOT_PROFILES = {
    'assembly_robot': RobotBehaviorProfile(
        robot_type='assembly_robot',
        common_actions=['read', 'execute'],
        common_resources=[
            'actuator_arm_01', 'actuator_arm_02',
            'sensor_position_01', 'sensor_force_01',
            'conveyor_belt_01',
        ],
        resource_types=['actuator', 'sensor', 'conveyor'],
        locations=['assembly_line', 'workstation_a'],
        request_rate=2.0,  # 2 requests/second
        pattern_variance=0.05,  # Very predictable
    ),
    
    'transport_robot': RobotBehaviorProfile(
        robot_type='transport_robot',
        common_actions=['read', 'write', 'execute'],
        common_resources=[
            'conveyor_belt_01', 'conveyor_belt_02',
            'database_inventory', 'sensor_weight_01',
        ],
        resource_types=['conveyor', 'database', 'sensor'],
        locations=['warehouse', 'assembly_line', 'loading_dock'],
        request_rate=1.5,
        pattern_variance=0.1,
    ),
    
    'camera_robot': RobotBehaviorProfile(
        robot_type='camera_robot',
        common_actions=['read'],
        common_resources=[
            'camera_01', 'camera_02',
            'database_vision', 'sensor_light_01',
        ],
        resource_types=['camera', 'database', 'sensor'],
        locations=['inspection_area', 'quality_control'],
        request_rate=3.0,  # High frequency
        pattern_variance=0.03,  # Very predictable
    ),
    
    'inspection_robot': RobotBehaviorProfile(
        robot_type='inspection_robot',
        common_actions=['read', 'execute'],
        common_resources=[
            'sensor_temperature_01', 'sensor_pressure_01',
            'database_maintenance', 'actuator_valve_01',
        ],
        resource_types=['sensor', 'database', 'actuator'],
        locations=['maintenance_bay', 'production_floor'],
        request_rate=1.0,
        pattern_variance=0.15,
    ),
}


class RobotAgentNode(Node):
    """
    ROS2 node simulating robot agent behavior.
    
    Publishes access requests and monitors decisions.
    """
    
    def __init__(self, robot_id: str, robot_type: str):
        """
        Initialize robot agent node.
        
        Args:
            robot_id: Unique robot identifier
            robot_type: Robot type (assembly, transport, camera, inspection)
        """
        super().__init__(f'robot_agent_{robot_id}')
        
        self.robot_id = robot_id
        self.robot_type = robot_type
        
        # Get behavior profile
        if robot_type not in ROBOT_PROFILES:
            raise ValueError(f"Unknown robot type: {robot_type}")
        
        self.profile = ROBOT_PROFILES[robot_type]
        
        self.get_logger().info(
            f"Robot agent initialized: {robot_id} ({robot_type})"
        )
        
        # Setup ROS communication
        self.request_pub = self.create_publisher(
            ROSAccessRequest,
            '/access_requests',
            10
        )
        
        self.decision_sub = self.create_subscription(
            ROSAccessDecision,
            '/access_decisions',
            self._decision_callback,
            10
        )
        
        # Statistics
        self.stats = {
            'requests_sent': 0,
            'grants_received': 0,
            'denials_received': 0,
            'approvals_required': 0,
        }
        
        # Timer for periodic requests
        period = 1.0 / self.profile.request_rate  # seconds
        self.timer = self.create_timer(period, self._send_request)
        
        self.get_logger().info(
            f"Publishing requests at {self.profile.request_rate} req/s"
        )
    
    def _decision_callback(self, msg: ROSAccessDecision):
        """Callback for access decisions."""
        # Only process decisions for this robot
        if not msg.request_id.startswith(self.robot_id):
            return
        
        # Update statistics
        decision = msg.decision.lower()
        
        if decision == 'grant':
            self.stats['grants_received'] += 1
        elif decision == 'deny':
            self.stats['denials_received'] += 1
        elif decision == 'require_approval':
            self.stats['approvals_required'] += 1
        
        self.get_logger().debug(
            f"Decision for {msg.request_id}: {decision} "
            f"(confidence={msg.confidence:.2f})"
        )
    
    def _send_request(self):
        """Send access request (called by timer)."""
        # Generate request
        request_id = f"{self.robot_id}_{int(time.time() * 1000000)}"
        
        request_data = self.profile.generate_request(
            self.robot_id,
            request_id
        )
        
        # Convert to ROS message
        msg = self._to_ros_message(request_data)
        
        # Publish
        self.request_pub.publish(msg)
        
        self.stats['requests_sent'] += 1
        
        if self.stats['requests_sent'] % 100 == 0:
            self._log_statistics()
    
    def _to_ros_message(self, data: Dict) -> ROSAccessRequest:
        """Convert request data to ROS message."""
        msg = ROSAccessRequest()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.request_id = data['request_id']
        msg.agent_id = data['agent_id']
        msg.agent_type = data['agent_type']
        msg.agent_role = data['agent_role']
        msg.action = data['action']
        msg.resource = data['resource']
        msg.resource_type = data['resource_type']
        msg.location = data['location']
        msg.human_present = data['human_present']
        msg.emergency = data['emergency']
        msg.session_id = data['session_id']
        msg.previous_action = data['previous_action']
        msg.auth_status = data['auth_status']
        msg.attempt_count = data['attempt_count']
        msg.policy_id = data['policy_id']
        msg.zone = data['zone']
        msg.priority = data['priority']
        
        return msg
    
    def _log_statistics(self):
        """Log agent statistics."""
        total = self.stats['requests_sent']
        grants = self.stats['grants_received']
        denials = self.stats['denials_received']
        approvals = self.stats['approvals_required']
        
        grant_rate = (grants / total * 100) if total > 0 else 0
        denial_rate = (denials / total * 100) if total > 0 else 0
        
        self.get_logger().info(
            f"Statistics: {total} requests sent | "
            f"Grants: {grant_rate:.1f}% | "
            f"Denials: {denial_rate:.1f}% | "
            f"Approvals: {approvals}"
        )
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        return self.stats.copy()


def main(args=None):
    """Main entry point."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: robot_agent.py <robot_id> <robot_type>")
        print("Example: robot_agent.py robot_assembly_01 assembly_robot")
        print("\nAvailable robot types:")
        for robot_type in ROBOT_PROFILES.keys():
            print(f"  - {robot_type}")
        sys.exit(1)
    
    robot_id = sys.argv[1]
    robot_type = sys.argv[2]
    
    rclpy.init(args=args)
    
    try:
        node = RobotAgentNode(robot_id, robot_type)
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down robot agent...")
        
        # Final statistics
        stats = node.get_statistics()
        node.get_logger().info("=" * 50)
        node.get_logger().info("FINAL STATISTICS")
        node.get_logger().info(f"Total requests: {stats['requests_sent']}")
        node.get_logger().info(f"Grants: {stats['grants_received']}")
        node.get_logger().info(f"Denials: {stats['denials_received']}")
        node.get_logger().info(f"Approvals: {stats['approvals_required']}")
        node.get_logger().info("=" * 50)
        
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()