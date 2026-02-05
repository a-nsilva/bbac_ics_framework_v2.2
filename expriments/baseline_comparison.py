#!/usr/bin/env python3
"""
BBAC ICS Framework - Baseline Comparison

Compares BBAC against traditional access control methods:
- RBAC (Role-Based Access Control)
- ABAC (Attribute-Based Access Control)
- Rule-based only
- Behavioral-only (no rules)
- BBAC (full hybrid)

Uses same test dataset for fair comparison.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node

from bbac_msgs.msg import AccessRequest as ROSAccessRequest
from bbac_msgs.msg import AccessDecision as ROSAccessDecision

from data.loader import DatasetLoader
from src.util.evaluator import MetricsEvaluator
from src.util.visualizer import Visualizer
from src.util.data_structures import (
    AccessRequest,
    AgentType,
    ActionType,
    ResourceType,
)


logger = logging.getLogger(__name__)


class BaselineMethod:
    """Base class for baseline methods."""
    
    def __init__(self, name: str):
        self.name = name
    
    def decide(self, request: AccessRequest) -> Tuple[str, float]:
        """
        Make access decision.
        
        Args:
            request: AccessRequest
            
        Returns:
            Tuple of (decision, confidence)
        """
        raise NotImplementedError


class RBACBaseline(BaselineMethod):
    """
    Pure RBAC baseline.
    
    Simple role-based rules:
    - Supervisors: grant all
    - Operators: grant read/write
    - Technicians: grant read/execute
    - Robots: grant read/execute on assigned resources
    """
    
    def __init__(self):
        super().__init__('RBAC')
        
        # Define role permissions
        self.permissions = {
            'supervisor': {'read', 'write', 'execute', 'delete'},
            'operator': {'read', 'write', 'execute'},
            'technician': {'read', 'execute'},
            'assembly_robot': {'read', 'execute'},
            'transport_robot': {'read', 'write'},
            'camera_robot': {'read'},
            'inspection_robot': {'read'},
        }
    
    def decide(self, request: AccessRequest) -> Tuple[str, float]:
        """Make RBAC decision."""
        role = request.agent_role.value.lower()
        action = request.action.value.lower()
        
        # Get allowed actions for role
        allowed_actions = self.permissions.get(role, set())
        
        if action in allowed_actions:
            return 'grant', 1.0
        else:
            return 'deny', 1.0


class ABACBaseline(BaselineMethod):
    """
    Attribute-Based Access Control baseline.
    
    Considers:
    - Role + Action (like RBAC)
    - Resource type
    - Context (human_present, emergency)
    - Location
    """
    
    def __init__(self):
        super().__init__('ABAC')
    
    def decide(self, request: AccessRequest) -> Tuple[str, float]:
        """Make ABAC decision."""
        # Emergency override
        if request.emergency:
            return 'grant', 1.0
        
        # Critical actions require human supervision for robots
        if request.agent_type == AgentType.ROBOT:
            if request.action in [ActionType.DELETE, ActionType.OVERRIDE]:
                if not request.human_present:
                    return 'deny', 0.9
        
        # Resource-based rules
        sensitive_resources = {'database', 'admin_panel', 'safety_system'}
        resource_type_lower = request.resource_type.value.lower()
        
        if any(sens in resource_type_lower for sens in sensitive_resources):
            # Sensitive resources - only humans with supervisor/operator roles
            if request.agent_type == AgentType.HUMAN:
                role = request.agent_role.value.lower()
                if 'supervisor' in role or 'operator' in role:
                    return 'grant', 0.8
            return 'deny', 0.8
        
        # Default: grant with moderate confidence
        return 'grant', 0.6


class RuleOnlyBaseline(BaselineMethod):
    """
    Pure rule-based (like PolicyEngine without behavioral/ML).
    """
    
    def __init__(self):
        super().__init__('Rule-Only')
    
    def decide(self, request: AccessRequest) -> Tuple[str, float]:
        """Make rule-based decision."""
        # Emergency
        if request.emergency:
            return 'grant', 1.0
        
        # Auth failures
        if request.attempt_count > 3:
            return 'deny', 1.0
        
        # Human supervision requirement
        if request.agent_type == AgentType.ROBOT:
            if request.action in [ActionType.DELETE, ActionType.OVERRIDE]:
                if not request.human_present:
                    return 'deny', 0.95
        
        # Supervisor override
        if request.agent_type == AgentType.HUMAN:
            role = request.agent_role.value.lower()
            if 'supervisor' in role:
                return 'grant', 0.9
        
        # Default: grant
        return 'grant', 0.5


class BehavioralOnlyBaseline(BaselineMethod):
    """
    Pure behavioral (statistical + sequence) without rules.
    
    Simple heuristic: grant if request matches common patterns.
    """
    
    def __init__(self):
        super().__init__('Behavioral-Only')
        
        # Simple pattern matching
        self.common_patterns = {
            'robot': {
                'actions': {'read', 'execute'},
                'resources': {'actuator', 'sensor', 'conveyor'},
            },
            'human': {
                'actions': {'read', 'write', 'execute'},
                'resources': {'database', 'admin_panel'},
            }
        }
    
    def decide(self, request: AccessRequest) -> Tuple[str, float]:
        """Make behavioral decision."""
        agent_type = request.agent_type.value.lower()
        action = request.action.value.lower()
        resource_type = request.resource_type.value.lower()
        
        patterns = self.common_patterns.get(agent_type, {})
        
        # Check if action is common
        action_match = action in patterns.get('actions', set())
        
        # Check if resource type is common
        resource_match = any(
            r in resource_type 
            for r in patterns.get('resources', set())
        )
        
        # Score based on matches
        if action_match and resource_match:
            return 'grant', 0.8
        elif action_match or resource_match:
            return 'grant', 0.6
        else:
            return 'deny', 0.5


class BaselineComparisonNode(Node):
    """
    ROS2 node for baseline comparison experiment.
    
    Runs BBAC and traditional baselines on same test data.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize baseline comparison node.
        
        Args:
            output_dir: Directory for results
        """
        super().__init__('baseline_comparison_node')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup evaluator and visualizer
        self.evaluator = MetricsEvaluator()
        self.visualizer = Visualizer(self.output_dir / 'figures')
        
        # Initialize baseline methods
        self.baselines = {
            'RBAC': RBACBaseline(),
            'ABAC': ABACBaseline(),
            'Rule-Only': RuleOnlyBaseline(),
            'Behavioral-Only': BehavioralOnlyBaseline(),
        }
        
        # Load dataset
        self.get_logger().info("Loading dataset...")
        self.dataset_loader = DatasetLoader()
        self.dataset_loader.load_all()
        
        self.test_data = self.dataset_loader.test_data
        
        if self.test_data is None or self.test_data.empty:
            raise RuntimeError("No test data available")
        
        self.get_logger().info(f"Loaded {len(self.test_data)} test samples")
        
        # Setup ROS communication (for BBAC)
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
        
        self.pending_requests = {}
        
        self.get_logger().info("BaselineComparisonNode initialized")
    
    def _decision_callback(self, msg: ROSAccessDecision):
        """Callback for BBAC decisions."""
        request_id = msg.request_id
        
        if request_id in self.pending_requests:
            self.pending_requests[request_id]['decision'] = msg.decision
            self.pending_requests[request_id]['confidence'] = msg.confidence
            self.pending_requests[request_id]['latency_ms'] = msg.latency_ms
            self.pending_requests[request_id]['received'] = True
    
    def run_baseline_method(
        self,
        method: BaselineMethod,
        requests: List[AccessRequest],
        ground_truth: List[str]
    ) -> Dict:
        """
        Run a baseline method.
        
        Args:
            method: Baseline method
            requests: List of requests
            ground_truth: Ground truth labels
            
        Returns:
            Results dictionary
        """
        self.get_logger().info(f"Running {method.name}...")
        
        predictions = []
        confidences = []
        latencies = []
        
        start_time = time.time()
        
        for request in requests:
            # Make decision
            decision_start = time.time()
            decision, confidence = method.decide(request)
            latency = (time.time() - decision_start) * 1000
            
            predictions.append(decision)
            confidences.append(confidence)
            latencies.append(latency)
        
        total_time = time.time() - start_time
        
        # Evaluate
        classification_metrics = self.evaluator.evaluate_classification(
            ground_truth,
            predictions,
            confidences
        )
        
        performance_metrics = self.evaluator.evaluate_performance(
            latencies,
            total_time
        )
        
        results = {
            'method': method.name,
            'classification': classification_metrics.to_dict(),
            'performance': performance_metrics.to_dict(),
            'predictions': predictions,
            'confidences': confidences,
            'latencies': latencies,
        }
        
        self.get_logger().info(
            f"{method.name}: F1={classification_metrics.f1:.4f}, "
            f"Latency={performance_metrics.latency.mean:.2f}ms"
        )
        
        return results
    
    def run_bbac_method(
        self,
        requests: List[AccessRequest],
        ground_truth: List[str]
    ) -> Dict:
        """
        Run BBAC (via ROS).
        
        Args:
            requests: List of requests
            ground_truth: Ground truth labels
            
        Returns:
            Results dictionary
        """
        self.get_logger().info("Running BBAC (via ROS)...")
        
        predictions = []
        confidences = []
        latencies = []
        
        start_time = time.time()
        
        for i, request in enumerate(requests):
            # Convert to ROS message
            ros_msg = self._to_ros_message(request)
            
            # Store pending
            self.pending_requests[request.request_id] = {
                'received': False,
            }
            
            # Publish
            self.request_pub.publish(ros_msg)
            
            # Wait for response
            timeout = 5.0
            start_wait = time.time()
            
            while not self.pending_requests[request.request_id]['received']:
                rclpy.spin_once(self, timeout_sec=0.01)
                
                if time.time() - start_wait > timeout:
                    self.get_logger().warning(f"Timeout for request {request.request_id}")
                    break
            
            # Collect result
            if self.pending_requests[request.request_id]['received']:
                predictions.append(self.pending_requests[request.request_id]['decision'])
                confidences.append(self.pending_requests[request.request_id]['confidence'])
                latencies.append(self.pending_requests[request.request_id]['latency_ms'])
            else:
                predictions.append('deny')
                confidences.append(0.0)
                latencies.append(timeout * 1000)
            
            if (i + 1) % 100 == 0:
                self.get_logger().info(f"Processed {i + 1}/{len(requests)} requests")
        
        total_time = time.time() - start_time
        
        # Evaluate
        classification_metrics = self.evaluator.evaluate_classification(
            ground_truth[:len(predictions)],
            predictions,
            confidences
        )
        
        performance_metrics = self.evaluator.evaluate_performance(
            latencies,
            total_time
        )
        
        results = {
            'method': 'BBAC',
            'classification': classification_metrics.to_dict(),
            'performance': performance_metrics.to_dict(),
            'predictions': predictions,
            'confidences': confidences,
            'latencies': latencies,
        }
        
        self.get_logger().info(
            f"BBAC: F1={classification_metrics.f1:.4f}, "
            f"Latency={performance_metrics.latency.mean:.2f}ms"
        )
        
        return results
    
    def _to_ros_message(self, request: AccessRequest) -> ROSAccessRequest:
        """Convert AccessRequest to ROS message."""
        msg = ROSAccessRequest()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.request_id = request.request_id
        msg.agent_id = request.agent_id
        msg.agent_type = request.agent_type.value
        msg.agent_role = request.agent_role.value
        msg.action = request.action.value
        msg.resource = request.resource
        msg.resource_type = request.resource_type.value
        msg.location = request.location
        msg.human_present = request.human_present
        msg.emergency = request.emergency
        msg.session_id = request.session_id if request.session_id else ''
        msg.previous_action = request.previous_action.value if request.previous_action else ''
        msg.auth_status = request.auth_status.value
        msg.attempt_count = request.attempt_count
        msg.policy_id = request.policy_id if request.policy_id else ''
        msg.zone = request.zone if request.zone else ''
        msg.priority = request.priority
        
        return msg
    
    def run_comparison(self, max_samples: int = 1000):
        """Run complete baseline comparison."""
        self.get_logger().info("=" * 70)
        self.get_logger().info("BASELINE COMPARISON STUDY")
        self.get_logger().info("=" * 70)
        
        # Prepare data
        test_samples = self.test_data.head(max_samples)
        
        requests = []
        ground_truth = []
        
        for idx, row in test_samples.iterrows():
            try:
                request = self.dataset_loader.to_access_request(row)
                requests.append(request)
                
                # Ground truth
                if 'expected_decision' in row:
                    gt = row['expected_decision']
                elif 'is_anomaly' in row:
                    gt = 'deny' if row['is_anomaly'] else 'grant'
                else:
                    gt = 'grant'
                
                ground_truth.append(gt)
                
            except Exception as e:
                self.get_logger().error(f"Error converting row {idx}: {e}")
                continue
        
        self.get_logger().info(f"Prepared {len(requests)} requests")
        
        # Run all methods
        all_results = {}
        
        # Run traditional baselines (local, fast)
        for name, method in self.baselines.items():
            results = self.run_baseline_method(method, requests, ground_truth)
            all_results[name] = results
        
        # Run BBAC (via ROS)
        bbac_results = self.run_bbac_method(requests, ground_truth)
        all_results['BBAC'] = bbac_results
        
        # Statistical comparison
        self._statistical_comparison(all_results, ground_truth)
        
        # Generate visualizations
        self._generate_visualizations(all_results)
        
        # Save results
        self._save_results(all_results)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("BASELINE COMPARISON COMPLETE")
        self.get_logger().info("=" * 70)
    
    def _statistical_comparison(self, all_results: Dict, ground_truth: List[str]):
        """Perform statistical comparison between methods."""
        from src.util.data_structures import ClassificationMetrics
        
        self.get_logger().info("Performing statistical tests...")
        
        # Compare BBAC vs each baseline
        bbac_results = all_results['BBAC']
        bbac_metrics = ClassificationMetrics(**bbac_results['classification'])
        bbac_latencies = bbac_results['latencies']
        
        comparisons = {}
        
        for method_name, results in all_results.items():
            if method_name == 'BBAC':
                continue
            
            method_metrics = ClassificationMetrics(**results['classification'])
            method_latencies = results['latencies']
            
            comparison = self.evaluator.compare_methods(
                bbac_metrics,
                method_metrics,
                bbac_latencies,
                method_latencies
            )
            
            comparisons[f'BBAC_vs_{method_name}'] = comparison
            
            # Log results
            self.get_logger().info(
                f"BBAC vs {method_name}: "
                f"F1 p-value={comparison['f1'].p_value:.4f} "
                f"({'significant' if comparison['f1'].significant else 'not significant'}), "
                f"Latency p-value={comparison['latency'].p_value:.4f}"
            )
        
        # Save comparisons
        comparison_file = self.output_dir / 'statistical_comparisons.json'
        
        with open(comparison_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in comparisons.items()},
                f,
                indent=2
            )
        
        self.get_logger().info(f"Saved statistical comparisons: {comparison_file}")
    
    def _generate_visualizations(self, all_results: Dict):
        """Generate comparison visualizations."""
        from src.util.data_structures import ClassificationMetrics
        
        # Extract metrics
        metrics_dict = {}
        latencies_dict = {}
        
        for method_name, results in all_results.items():
            cm_dict = results['classification']
            metrics_dict[method_name] = ClassificationMetrics(**cm_dict)
            latencies_dict[method_name] = results['latencies']
        
        # Plot ROC curves
        self.visualizer.plot_roc_curve(
            metrics_dict,
            title='ROC Curve - Baseline Comparison',
            filename='roc_baseline_comparison.png'
        )
        
        # Plot PR curves
        self.visualizer.plot_precision_recall_curve(
            metrics_dict,
            title='Precision-Recall Curve - Baseline Comparison',
            filename='pr_baseline_comparison.png'
        )
        
        # Plot metrics comparison
        self.visualizer.plot_metrics_comparison(
            metrics_dict,
            title='Metrics Comparison - Baseline Methods',
            filename='metrics_baseline_comparison.png'
        )
        
        # Plot latency distributions
        self.visualizer.plot_latency_distribution(
            latencies_dict,
            title='Latency Distribution - Baseline Comparison',
            filename='latency_baseline_comparison.png'
        )
        
        self.get_logger().info("Generated all visualizations")
    
    def _save_results(self, all_results: Dict):
        """Save all results."""
        # Save individual results
        for method_name, results in all_results.items():
            output_file = self.output_dir / f'{method_name}_results.json'
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / 'comparison_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("BASELINE COMPARISON SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for method_name, results in all_results.items():
                f.write(f"{method_name}\n")
                f.write("-" * 50 + "\n")
                
                cm = results['classification']
                perf = results['performance']
                
                f.write(f"Accuracy:  {cm['accuracy']:.4f}\n")
                f.write(f"Precision: {cm['precision']:.4f}\n")
                f.write(f"Recall:    {cm['recall']:.4f}\n")
                f.write(f"F1 Score:  {cm['f1_score']:.4f}\n")
                f.write(f"ROC-AUC:   {cm['roc_auc']:.4f}\n")
                f.write(f"\n")
                f.write(f"Latency (mean): {perf['latency_stats']['mean']:.2f} ms\n")
                f.write(f"Latency (p95):  {perf['latency_stats']['p95']:.2f} ms\n")
                f.write(f"Latency (p99):  {perf['latency_stats']['p99']:.2f} ms\n")
                f.write(f"Throughput:     {perf['throughput']:.2f} req/s\n")
                f.write("\n\n")
        
        self.get_logger().info(f"Saved results to {self.output_dir}")


def main(args=None):
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    rclpy.init(args=args)
    
    output_dir = Path('results/baseline_comparison')
    node = BaselineComparisonNode(output_dir)
    
    try:
        node.run_comparison(max_samples=1000)
        
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        
    except Exception as e:
        node.get_logger().error(f"Error in baseline comparison: {e}")
        raise
        
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()