#!/usr/bin/env python3
"""
BBAC ICS Framework - Human Agent Simulator

Simulates human agents publishing access requests to BBAC node.
Generates more variable behavioral patterns compared to robots.
"""

import json
import logging
import random
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node

from bbac_framework.msg import AccessDecision as ROSAccessDecision
from bbac_framework.msg import AccessRequest as ROSAccessRequest


logger = logging.getLogger(__name__)


class HumanBehaviorProfile:
    """Defines behavioral pattern for a human role."""
    
    def __init__(
        self,
        role: str,
        common_actions: List[str],
        common_resources: List[str],
        resource_types: List[str],
        locations: List[str],
        request_rate: float = 0.5,  # Lower than robots
        pattern_variance: float = 0.3,  # Higher variance
        shift_hours: tuple = (8, 17),  # Work hours
    ):
        """
        Initialize human behavior profile.
        
        Args:
            role: Human role (supervisor, operator, technician)
            common_actions: List of common actions
            common_resources: List of common resources
            resource_types: List of resource types
            locations: Typical locations
            request_rate: Requests per second
            pattern_variance: Variance in behavior (humans are less predictable)
            shift_hours: Work shift hours (start, end)
        """
        self.role = role
        self.common_actions = common_actions
        self.common_resources = common_resources
        self.resource_types = resource_types
        self.locations = locations
        self.request_rate = request_rate
        self.pattern_variance = pattern_variance
        self.shift_hours = shift_hours
        
        self.last_action = 'read'
    
    def is_work_hours(self) -> bool:
        """Check if current time is within work hours."""
        current_hour = time.localtime().tm_hour
        start, end = self.shift_hours
        
        return start <= current_hour < end
    
    def generate_request(self, agent_id: str, request_id: str) -> Dict:
        """Generate access request based on behavior profile."""
        # Adjust request rate based on work hours
        if not self.is_work_hours():
            # Off-hours: much lower request rate, different patterns
            if random.random() > 0.1:  # 90% chance to skip
                return None
        
        # Action selection (more random than robots)
        if random.random() < self.pattern_variance:
            action = random.choice(['read', 'write', 'execute', 'delete', 'override'])
        else:
            action = random.choice(self.common_actions)
        
        self.last_action = action
        
        # Resource selection
        if random.random() < self.pattern_variance:
            resource_type = random.choice(self.resource_types)
            resource = f"{resource_type}_custom_{random.randint(1, 50)}"
        else:
            resource = random.choice(self.common_resources)
            resource_type = self.resource_types[0]  # Default
        
        # Location (humans move around more)
        if random.random() < 0.2:  # 20% chance of different location
            location = random.choice(['office', 'control_room', 'cafeteria'])
        else:
            location = random.choice(self.locations)
        
        # Context
        human_present = True  # Obviously
        emergency = random.random() < 0.02  # 2% chance (humans handle emergencies)
        
        # Auth (humans occasionally have auth issues)
        if random.random() < 0.05:  # 5% chance
            auth_status = 'failed'
            attempt_count = random.randint(1, 3)
        else:
            auth_status = 'success'
            attempt_count = 0
        
        return {
            'request_id': request_id,
            'agent_id': agent_id,
            'agent_type': 'human',
            'agent_role': self.role,
            'action': action,
            'resource': resource,
            'resource_type': resource_type,
            'location': location,
            'human_present': human_present,
            'emergency': emergency,
            'session_id': f"session_{agent_id}_{int(time.time())}",
            'previous_action': self.last_action,
            'auth_status': auth_status,
            'attempt_count': attempt_count,
            'policy_id': '',
            'zone': location,
            'priority': 7.0 if emergency else 5.0,
        }


# Predefined human profiles
HUMAN_PROFILES = {
    'supervisor': HumanBehaviorProfile(
        role='supervisor',
        common_actions=['read', 'write', 'execute', 'delete', 'override'],
        common_resources=[
            'database_production', 'database_inventory',
            'admin_panel', 'safety_system',
            'actuator_emergency_stop',
        ],
        resource_types=['database', 'admin_panel', 'safety_system', 'actuator'],
        locations=['control_room', 'production_floor', 'office'],
        request_rate=0.3,
        pattern_variance=0.4,  # Supervisors are unpredictable
        shift_hours=(7, 19),  # Long shifts
    ),
    
    'operator': HumanBehaviorProfile(
        role='operator',
        common_actions=['read', 'write', 'execute'],
        common_resources=[
            'conveyor_belt_01', 'actuator_arm_01',
            'database_production', 'sensor_position_01',
        ],
        resource_types=['conveyor', 'actuator', 'database', 'sensor'],
        locations=['workstation_a', 'workstation_b', 'control_room'],
        request_rate=0.5,
        pattern_variance=0.3,
        shift_hours=(8, 17),
    ),
    
    'technician': HumanBehaviorProfile(
        role='technician',
        common_actions=['read', 'execute', 'write'],
        common_resources=[
            'actuator_arm_01', 'sensor_temperature_01',
            'database_maintenance', 'conveyor_belt_01',
        ],
        resource_types=['actuator', 'sensor', 'database', 'conveyor'],
        locations=['maintenance_bay', 'production_floor', 'warehouse'],
        request_rate=0.4,
        pattern_variance=0.35,
        shift_hours=(6, 15),  # Early shift
    ),
}


class HumanAgentNode(Node):
    """
    ROS2 node simulating human agent behavior.
    
    More variable patterns than robots, work-hour awareness.
    """
    
    def __init__(self, human_id: str, role: str):
        """
        Initialize human agent node.
        
        Args:
            human_id: Unique human identifier
            role: Human role (supervisor, operator, technician)
        """
        super().__init__(f'human_agent_{human_id}')
        
        self.human_id = human_id
        self.role = role
        
        # Get behavior profile
        if role not in HUMAN_PROFILES:
            raise ValueError(f"Unknown human role: {role}")
        
        self.profile = HUMAN_PROFILES[role]
        
        self.get_logger().info(
            f"Human agent initialized: {human_id} ({role})"
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
            'auth_failures': 0,
        }
        
        # Timer for periodic requests
        period = 1.0 / self.profile.request_rate
        self.timer = self.create_timer(period, self._send_request)
        
        self.get_logger().info(
            f"Publishing requests at {self.profile.request_rate} req/s "
            f"(shift: {self.profile.shift_hours[0]}-{self.profile.shift_hours[1]}h)"
        )
    
    def _decision_callback(self, msg: ROSAccessDecision):
        """Callback for access decisions."""
        if not msg.request_id.startswith(self.human_id):
            return
        
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
        request_id = f"{self.human_id}_{int(time.time() * 1000000)}"
        
        request_data = self.profile.generate_request(
            self.human_id,
            request_id
        )
        
        # Skip if None (off-hours)
        if request_data is None:
            return
        
        # Track auth failures
        if request_data['auth_status'] != 'success':
            self.stats['auth_failures'] += 1
        
        # Convert to ROS message
        msg = self._to_ros_message(request_data)
        
        # Publish
        self.request_pub.publish(msg)
        
        self.stats['requests_sent'] += 1
        
        if self.stats['requests_sent'] % 50 == 0:
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
        
        grant_rate = (grants / total * 100) if total > 0 else 0
        denial_rate = (denials / total * 100) if total > 0 else 0
        
        work_hours_status = "ON" if self.profile.is_work_hours() else "OFF"
        
        self.get_logger().info(
            f"[{work_hours_status} shift] {total} requests | "
            f"Grants: {grant_rate:.1f}% | Denials: {denial_rate:.1f}% | "
            f"Auth failures: {self.stats['auth_failures']}"
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
        print("Usage: human_agent.py <human_id> <role>")
        print("Example: human_agent.py human_supervisor_01 supervisor")
        print("\nAvailable roles:")
        for role in HUMAN_PROFILES.keys():
            print(f"  - {role}")
        sys.exit(1)
    
    human_id = sys.argv[1]
    role = sys.argv[2]
    
    rclpy.init(args=args)
    
    try:
        node = HumanAgentNode(human_id, role)
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down human agent...")
        
        # Final statistics
        stats = node.get_statistics()
        node.get_logger().info("=" * 50)
        node.get_logger().info("FINAL STATISTICS")
        node.get_logger().info(f"Total requests: {stats['requests_sent']}")
        node.get_logger().info(f"Grants: {stats['grants_received']}")
        node.get_logger().info(f"Denials: {stats['denials_received']}")
        node.get_logger().info(f"Auth failures: {stats['auth_failures']}")
        node.get_logger().info("=" * 50)
        
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()