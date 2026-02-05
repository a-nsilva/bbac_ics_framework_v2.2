#!/usr/bin/env python3
"""
BBAC ICS Framework - Ablation Study

Tests each layer individually vs hybrid approach:
- Rule-only (RuBAC)
- Statistical-only
- Sequence-only (Markov)
- Statistical + Sequence
- Full hybrid (all layers)

Publishes requests to ROS /access_requests and collects decisions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from bbac_msgs.msg import AccessRequest as ROSAccessRequest
from bbac_msgs.msg import AccessDecision as ROSAccessDecision

from data.loader import DatasetLoader
from src.util.evaluator import MetricsEvaluator
from src.util.visualizer import Visualizer
from src.util.config_loader import config
from src.util.data_structures import AccessRequest


logger = logging.getLogger(__name__)


class AblationStudyNode(Node):
    """
    ROS2 node for ablation study experiment.
    
    Tests different layer combinations by dynamically reconfiguring
    the BBAC node via parameters.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize ablation study node.
        
        Args:
            output_dir: Directory for results
        """
        super().__init__('ablation_study_node')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup evaluator and visualizer
        self.evaluator = MetricsEvaluator()
        self.visualizer = Visualizer(self.output_dir / 'figures')
        
        # Load dataset
        self.get_logger().info("Loading dataset...")
        self.dataset_loader = DatasetLoader()
        self.dataset_loader.load_all()
        
        # Get test data
        self.test_data = self.dataset_loader.test_data
        
        if self.test_data is None or self.test_data.empty:
            raise RuntimeError("No test data available")
        
        self.get_logger().info(f"Loaded {len(self.test_data)} test samples")
        
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
        
        # Results storage
        self.pending_requests = {}
        self.results = {}
        
        self.get_logger().info("AblationStudyNode initialized")
    
    def _decision_callback(self, msg: ROSAccessDecision):
        """Callback for receiving decisions."""
        request_id = msg.request_id
        
        if request_id in self.pending_requests:
            decision = msg.decision
            confidence = msg.confidence
            latency = msg.latency_ms
            
            self.pending_requests[request_id]['decision'] = decision
            self.pending_requests[request_id]['confidence'] = confidence
            self.pending_requests[request_id]['latency_ms'] = latency
            self.pending_requests[request_id]['received'] = True
    
    def run_configuration(
        self,
        config_name: str,
        enable_behavioral: bool,
        enable_ml: bool,
        enable_policy: bool,
        max_samples: int = 1000
    ) -> Dict:
        """
        Run experiment with specific configuration.
        
        Args:
            config_name: Configuration name
            enable_behavioral: Enable statistical layer
            enable_ml: Enable ML/sequence layer
            enable_policy: Enable policy layer
            max_samples: Maximum number of samples to test
            
        Returns:
            Results dictionary
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Running configuration: {config_name}")
        self.get_logger().info(
            f"Layers: behavioral={enable_behavioral}, "
            f"ml={enable_ml}, policy={enable_policy}"
        )
        self.get_logger().info("=" * 70)
        
        # TODO: Set BBAC node parameters dynamically
        # For now, assume BBAC node is restarted with correct params
        
        # Prepare data
        test_samples = self.test_data.head(max_samples)
        
        # Convert to requests
        requests = []
        ground_truth = []
        
        for idx, row in test_samples.iterrows():
            try:
                request = self.dataset_loader.to_access_request(row)
                requests.append(request)
                
                # Ground truth (assume 'is_anomaly' or 'expected_decision' column)
                if 'expected_decision' in row:
                    gt = row['expected_decision']
                elif 'is_anomaly' in row:
                    gt = 'deny' if row['is_anomaly'] else 'grant'
                else:
                    # Default heuristic
                    gt = 'grant'
                
                ground_truth.append(gt)
                
            except Exception as e:
                self.get_logger().error(f"Error converting row {idx}: {e}")
                continue
        
        self.get_logger().info(f"Prepared {len(requests)} requests")
        
        # Send requests and collect decisions
        predictions = []
        confidences = []
        latencies = []
        
        start_time = time.time()
        
        for i, request in enumerate(requests):
            # Convert to ROS message
            ros_msg = self._to_ros_message(request)
            
            # Store pending
            self.pending_requests[request.request_id] = {
                'request': request,
                'ground_truth': ground_truth[i],
                'received': False,
            }
            
            # Publish
            self.request_pub.publish(ros_msg)
            
            # Wait for response (with timeout)
            timeout = 5.0  # 5 seconds
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
                # Timeout - assume deny
                predictions.append('deny')
                confidences.append(0.0)
                latencies.append(timeout * 1000)
            
            # Progress
            if (i + 1) % 100 == 0:
                self.get_logger().info(f"Processed {i + 1}/{len(requests)} requests")
        
        total_time = time.time() - start_time
        
        self.get_logger().info(f"Completed in {total_time:.2f}s")
        
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
            'config_name': config_name,
            'enable_behavioral': enable_behavioral,
            'enable_ml': enable_ml,
            'enable_policy': enable_policy,
            'classification': classification_metrics.to_dict(),
            'performance': performance_metrics.to_dict(),
            'predictions': predictions,
            'ground_truth': ground_truth[:len(predictions)],
            'confidences': confidences,
            'latencies': latencies,
        }
        
        # Save results
        self._save_results(config_name, results)
        
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
    
    def _save_results(self, config_name: str, results: Dict):
        """Save results to JSON."""
        output_file = self.output_dir / f'{config_name}_results.json'
        
        # Convert non-serializable objects
        results_serializable = results.copy()
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            self.get_logger().info(f"Saved results: {output_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving results: {e}")
    
    def run_all_configurations(self, max_samples: int = 1000):
        """Run all ablation study configurations."""
        configurations = [
            ('rule_only', False, False, True),
            ('statistical_only', True, False, False),
            ('sequence_only', False, True, False),
            ('stat_seq', True, True, False),
            ('full_hybrid', True, True, True),
        ]
        
        all_results = {}
        
        for config_name, behavioral, ml, policy in configurations:
            results = self.run_configuration(
                config_name,
                behavioral,
                ml,
                policy,
                max_samples
            )
            
            all_results[config_name] = results
            
            # Wait between configurations
            time.sleep(2.0)
        
        # Generate comparison visualizations
        self._generate_visualizations(all_results)
        
        # Save summary
        self._save_summary(all_results)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("ABLATION STUDY COMPLETE")
        self.get_logger().info("=" * 70)
    
    def _generate_visualizations(self, all_results: Dict):
        """Generate comparison visualizations."""
        from src.util.data_structures import ClassificationMetrics
        
        # Extract metrics
        metrics_dict = {}
        latencies_dict = {}
        
        for config_name, results in all_results.items():
            # Reconstruct ClassificationMetrics
            cm_dict = results['classification']
            metrics_dict[config_name] = ClassificationMetrics(**cm_dict)
            
            latencies_dict[config_name] = results['latencies']
        
        # Plot confusion matrices
        for config_name, metrics in metrics_dict.items():
            self.visualizer.plot_confusion_matrix(
                metrics,
                title=f'Confusion Matrix - {config_name}',
                filename=f'cm_{config_name}.png'
            )
        
        # Plot ROC curves
        self.visualizer.plot_roc_curve(
            metrics_dict,
            title='ROC Curve - Ablation Study',
            filename='roc_ablation.png'
        )
        
        # Plot metrics comparison
        self.visualizer.plot_metrics_comparison(
            metrics_dict,
            title='Metrics Comparison - Ablation Study',
            filename='metrics_ablation.png'
        )
        
        # Plot latency distributions
        self.visualizer.plot_latency_distribution(
            latencies_dict,
            title='Latency Distribution - Ablation Study',
            filename='latency_ablation.png'
        )
        
        self.get_logger().info("Generated all visualizations")
    
    def _save_summary(self, all_results: Dict):
        """Save summary report."""
        summary_file = self.output_dir / 'ablation_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ABLATION STUDY SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for config_name, results in all_results.items():
                f.write(f"{config_name.upper()}\n")
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
        
        self.get_logger().info(f"Saved summary: {summary_file}")


def main(args=None):
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    rclpy.init(args=args)
    
    output_dir = Path('results/ablation_study')
    node = AblationStudyNode(output_dir)
    
    try:
        # Run all configurations
        node.run_all_configurations(max_samples=1000)
        
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        
    except Exception as e:
        node.get_logger().error(f"Error in ablation study: {e}")
        raise
        
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()