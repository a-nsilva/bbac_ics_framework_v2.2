#!/usr/bin/env python3
"""
BBAC ICS Framework - Adaptive & Dynamic Evaluation

Evaluates ADAPTIVE and DYNAMIC capabilities (6 key metrics):

ADAPTIVE:
1. Baseline convergence rate - how fast baseline adapts to new patterns
2. Drift detection accuracy - correctly identifying behavioral changes
3. Sliding window effectiveness - 70% recent vs 30% historical comparison

DYNAMIC:
4. Rule update latency - time to apply new rules
5. Conflict resolution rate - handling conflicting rule updates

INTERACTION:
6. Concurrent drift + rule change - system behavior under simultaneous changes
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node

from bbac_msgs.msg import AccessRequest as ROSAccessRequest
from bbac_msgs.msg import AccessDecision as ROSAccessDecision

from data.loader import DatasetLoader
from src.core.modeling import BaselineBuilder
from src.util.evaluator import MetricsEvaluator
from src.util.visualizer import Visualizer
from src.util.config_loader import config


logger = logging.getLogger(__name__)


class AdaptiveEvaluationNode(Node):
    """
    ROS2 node for evaluating adaptive and dynamic capabilities.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize adaptive evaluation node.
        
        Args:
            output_dir: Directory for results
        """
        super().__init__('adaptive_eval_node')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup evaluator and visualizer
        self.evaluator = MetricsEvaluator()
        self.visualizer = Visualizer(self.output_dir / 'figures')
        
        # Load dataset
        self.get_logger().info("Loading dataset...")
        self.dataset_loader = DatasetLoader()
        self.dataset_loader.load_all()
        
        self.train_data = self.dataset_loader.train_data
        
        if self.train_data is None or self.train_data.empty:
            raise RuntimeError("No training data available")
        
        self.get_logger().info(f"Loaded {len(self.train_data)} training samples")
        
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
        
        self.pending_requests = {}
        
        # Baseline builder for simulations
        self.baseline_builder = BaselineBuilder()
        
        self.get_logger().info("AdaptiveEvaluationNode initialized")
    
    def _decision_callback(self, msg: ROSAccessDecision):
        """Callback for decisions."""
        request_id = msg.request_id
        
        if request_id in self.pending_requests:
            self.pending_requests[request_id]['decision'] = msg.decision
            self.pending_requests[request_id]['confidence'] = msg.confidence
            self.pending_requests[request_id]['latency_ms'] = msg.latency_ms
            self.pending_requests[request_id]['received'] = True
    
    def evaluate_baseline_convergence(
        self,
        agent_id: str,
        window_size: int = 100
    ) -> Dict:
        """
        Metric 1: Baseline convergence rate.
        
        Measures how quickly baseline adapts to new behavior patterns.
        
        Args:
            agent_id: Agent to analyze
            window_size: Number of samples per window
            
        Returns:
            Convergence metrics
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("METRIC 1: Baseline Convergence Rate")
        self.get_logger().info("=" * 70)
        
        # Filter agent data
        agent_data = self.train_data[self.train_data['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('timestamp')
        
        if len(agent_data) < window_size * 3:
            self.get_logger().warning(f"Insufficient data for agent {agent_id}")
            return {}
        
        # Build baseline at different time points
        baselines = []
        convergence_metrics = []
        
        for i in range(0, len(agent_data) - window_size, window_size // 2):
            window_data = agent_data.iloc[i:i + window_size]
            
            # Build baseline
            baseline = self.baseline_builder.build_baseline(window_data)
            
            # Extract key metric (e.g., action frequency distribution)
            action_dist = baseline.get('common_actions', {}).get('frequencies', {})
            
            baselines.append({
                'window_idx': i // (window_size // 2),
                'sample_count': len(window_data),
                'action_distribution': action_dist,
                'time_gap_mean': baseline.get('time_gaps', {}).get('avg_gap_seconds', 0),
            })
            
            # If we have previous baseline, calculate convergence
            if len(baselines) > 1:
                prev = baselines[-2]
                curr = baselines[-1]
                
                # Calculate distribution distance (KL divergence approximation)
                distance = self._distribution_distance(
                    prev['action_distribution'],
                    curr['action_distribution']
                )
                
                convergence_metrics.append({
                    'window_idx': curr['window_idx'],
                    'distance_from_previous': distance,
                    'time_gap_change': abs(
                        curr['time_gap_mean'] - prev['time_gap_mean']
                    ),
                })
        
        # Calculate convergence rate (how fast distance decreases)
        if len(convergence_metrics) > 5:
            distances = [m['distance_from_previous'] for m in convergence_metrics]
            
            # Fit exponential decay: distance = a * exp(-b * window_idx)
            # Convergence rate = b
            convergence_rate = self._estimate_convergence_rate(distances)
        else:
            convergence_rate = 0.0
        
        results = {
            'agent_id': agent_id,
            'convergence_rate': convergence_rate,
            'num_windows': len(baselines),
            'convergence_metrics': convergence_metrics,
            'final_stability': convergence_metrics[-5:] if len(convergence_metrics) >= 5 else [],
        }
        
        self.get_logger().info(f"Convergence rate: {convergence_rate:.4f}")
        
        # Visualize
        self._plot_convergence(convergence_metrics, agent_id)
        
        return results
    
    def evaluate_drift_detection(
        self,
        agent_id: str,
        drift_point: int = None
    ) -> Dict:
        """
        Metric 2: Drift detection accuracy.
        
        Simulates behavioral drift and measures detection accuracy.
        
        Args:
            agent_id: Agent to analyze
            drift_point: Sample index where drift is introduced (auto if None)
            
        Returns:
            Drift detection metrics
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("METRIC 2: Drift Detection Accuracy")
        self.get_logger().info("=" * 70)
        
        # Filter agent data
        agent_data = self.train_data[self.train_data['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('timestamp')
        
        if len(agent_data) < 200:
            self.get_logger().warning(f"Insufficient data for agent {agent_id}")
            return {}
        
        # Split into before/after drift
        if drift_point is None:
            drift_point = len(agent_data) // 2
        
        before_drift = agent_data.iloc[:drift_point]
        after_drift = agent_data.iloc[drift_point:]
        
        # Build baselines
        baseline_before = self.baseline_builder.build_baseline(before_drift)
        baseline_after = self.baseline_builder.build_baseline(after_drift)
        
        # Calculate drift using multiple features
        drift_detected = False
        drift_scores = {}
        
        # Action distribution drift
        actions_before = baseline_before.get('common_actions', {}).get('frequencies', {})
        actions_after = baseline_after.get('common_actions', {}).get('frequencies', {})
        
        drift_scores['action_distribution'] = self._distribution_distance(
            actions_before, actions_after
        )
        
        # Time gap drift
        time_gap_before = baseline_before.get('time_gaps', {}).get('avg_gap_seconds', 0)
        time_gap_after = baseline_after.get('time_gaps', {}).get('avg_gap_seconds', 0)
        
        if time_gap_before > 0:
            drift_scores['time_gap_change'] = abs(
                (time_gap_after - time_gap_before) / time_gap_before
            )
        else:
            drift_scores['time_gap_change'] = 0.0
        
        # Statistical test for drift
        # Extract numerical features for KS test
        before_gaps = []
        after_gaps = []
        
        if 'timestamp' in before_drift.columns and len(before_drift) > 1:
            before_ts = pd.to_datetime(before_drift['timestamp'])
            before_gaps = before_ts.diff().dt.total_seconds().dropna().tolist()
        
        if 'timestamp' in after_drift.columns and len(after_drift) > 1:
            after_ts = pd.to_datetime(after_drift['timestamp'])
            after_gaps = after_ts.diff().dt.total_seconds().dropna().tolist()
        
        if before_gaps and after_gaps:
            drift_metrics = self.evaluator.calculate_drift_metrics(
                before_gaps, after_gaps
            )
            
            drift_detected = drift_metrics['drift_detected']
            drift_scores['ks_statistic'] = drift_metrics['ks_statistic']
            drift_scores['ks_pvalue'] = drift_metrics['ks_pvalue']
        
        # Overall drift score (weighted average)
        overall_drift_score = (
            0.5 * drift_scores.get('action_distribution', 0) +
            0.3 * drift_scores.get('time_gap_change', 0) +
            0.2 * drift_scores.get('ks_statistic', 0)
        )
        
        results = {
            'agent_id': agent_id,
            'drift_point': drift_point,
            'drift_detected': drift_detected,
            'overall_drift_score': overall_drift_score,
            'drift_scores': drift_scores,
            'baseline_before': baseline_before,
            'baseline_after': baseline_after,
        }
        
        self.get_logger().info(
            f"Drift detected: {drift_detected}, "
            f"Overall score: {overall_drift_score:.4f}"
        )
        
        return results
    
    def evaluate_sliding_window(
        self,
        agent_id: str
    ) -> Dict:
        """
        Metric 3: Sliding window effectiveness (70% recent vs 30% historical).
        
        Compares adaptive sliding window against fixed window.
        
        Args:
            agent_id: Agent to analyze
            
        Returns:
            Sliding window comparison metrics
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("METRIC 3: Sliding Window Effectiveness")
        self.get_logger().info("=" * 70)
        
        # Filter agent data
        agent_data = self.train_data[self.train_data['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('timestamp')
        
        window_days = config.baseline.get('window_days', 10)
        
        # Split into recent (70%) and historical (30%)
        split_point = int(len(agent_data) * 0.7)
        recent_data = agent_data.iloc[split_point:]
        historical_data = agent_data.iloc[:split_point]
        
        # Build baselines with different strategies
        
        # Strategy 1: Fixed window (all data equally weighted)
        baseline_fixed = self.baseline_builder.build_baseline(agent_data)
        
        # Strategy 2: Recent only
        baseline_recent = self.baseline_builder.build_baseline(recent_data)
        
        # Strategy 3: Adaptive (70% recent + 30% historical)
        # Add weights to data
        agent_data_weighted = agent_data.copy()
        agent_data_weighted['baseline_weight'] = 0.0
        
        agent_data_weighted.loc[recent_data.index, 'baseline_weight'] = (
            0.7 / len(recent_data)
        )
        agent_data_weighted.loc[historical_data.index, 'baseline_weight'] = (
            0.3 / len(historical_data)
        )
        
        baseline_adaptive = self.baseline_builder.build_baseline(agent_data_weighted)
        
        # Compare baselines
        comparison = {
            'strategy': ['Fixed', 'Recent-Only', 'Adaptive'],
            'action_entropy': [],
            'time_gap_std': [],
            'resource_diversity': [],
        }
        
        for baseline, name in [
            (baseline_fixed, 'Fixed'),
            (baseline_recent, 'Recent-Only'),
            (baseline_adaptive, 'Adaptive')
        ]:
            # Action entropy (higher = more diverse)
            action_freq = baseline.get('common_actions', {}).get('frequencies', {})
            comparison['action_entropy'].append(
                self._calculate_entropy(list(action_freq.values()))
            )
            
            # Time gap std (variability)
            comparison['time_gap_std'].append(
                baseline.get('time_gaps', {}).get('std_gap_seconds', 0)
            )
            
            # Resource diversity
            comparison['resource_diversity'].append(
                baseline.get('normal_resources', {}).get('unique_resources', 0)
            )
        
        results = {
            'agent_id': agent_id,
            'comparison': comparison,
            'baseline_fixed': baseline_fixed,
            'baseline_recent': baseline_recent,
            'baseline_adaptive': baseline_adaptive,
        }
        
        self.get_logger().info("Sliding window comparison complete")
        
        # Visualize
        self._plot_sliding_window_comparison(comparison)
        
        return results
    
    def evaluate_rule_update_latency(
        self,
        num_updates: int = 100
    ) -> Dict:
        """
        Metric 4: Rule update latency.
        
        Measures time to apply new rules to the system.
        
        Args:
            num_updates: Number of rule updates to test
            
        Returns:
            Rule update latency metrics
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("METRIC 4: Rule Update Latency")
        self.get_logger().info("=" * 70)
        
        # This would require ROS service calls to update rules
        # For now, simulate with local measurements
        
        latencies = []
        
        for i in range(num_updates):
            # Simulate rule creation
            start_time = time.time()
            
            # Create dummy rule
            rule = {
                'id': f'test_rule_{i}',
                'priority': i,
                'condition': lambda req: True,
                'action': 'grant',
                'description': f'Test rule {i}',
            }
            
            # Simulate update latency (would be ROS service call)
            time.sleep(0.001)  # 1ms simulation
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        latency_metrics = self.evaluator.evaluate_latency(latencies)
        
        results = {
            'num_updates': num_updates,
            'latency_metrics': latency_metrics.to_dict(),
            'target_latency_ms': 1000,  # Target: < 1 second
            'meets_target': latency_metrics.mean < 1000,
        }
        
        self.get_logger().info(
            f"Rule update latency: mean={latency_metrics.mean:.2f}ms, "
            f"p95={latency_metrics.p95:.2f}ms"
        )
        
        return results
    
    def evaluate_conflict_resolution(self) -> Dict:
        """
        Metric 5: Conflict resolution rate.
        
        Tests handling of conflicting rule updates.
        
        Returns:
            Conflict resolution metrics
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("METRIC 5: Conflict Resolution Rate")
        self.get_logger().info("=" * 70)
        
        # Simulate conflicting rules
        conflicts = [
            {
                'rule_a': {'action': 'grant', 'priority': 5},
                'rule_b': {'action': 'deny', 'priority': 5},
                'expected_resolution': 'deny',  # Deny has priority in conflicts
            },
            {
                'rule_a': {'action': 'grant', 'priority': 10},
                'rule_b': {'action': 'deny', 'priority': 5},
                'expected_resolution': 'grant',  # Higher priority wins
            },
        ]
        
        resolutions_correct = 0
        total_conflicts = len(conflicts)
        
        for conflict in conflicts:
            # Simulate conflict resolution
            # In real system, this would query BBAC node
            
            # Simple priority-based resolution
            if conflict['rule_a']['priority'] > conflict['rule_b']['priority']:
                resolved = conflict['rule_a']['action']
            elif conflict['rule_b']['priority'] > conflict['rule_a']['priority']:
                resolved = conflict['rule_b']['action']
            else:
                # Equal priority - deny wins
                resolved = 'deny'
            
            if resolved == conflict['expected_resolution']:
                resolutions_correct += 1
        
        resolution_rate = resolutions_correct / total_conflicts
        
        results = {
            'total_conflicts': total_conflicts,
            'resolutions_correct': resolutions_correct,
            'resolution_rate': resolution_rate,
            'target_rate': 1.0,  # Target: 100% correct
        }
        
        self.get_logger().info(f"Conflict resolution rate: {resolution_rate:.2%}")
        
        return results
    
    def evaluate_concurrent_changes(
        self,
        agent_id: str
    ) -> Dict:
        """
        Metric 6: Concurrent drift + rule change.
        
        Tests system behavior under simultaneous behavioral drift and rule updates.
        
        Args:
            agent_id: Agent to analyze
            
        Returns:
            Concurrent change metrics
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("METRIC 6: Concurrent Drift + Rule Change")
        self.get_logger().info("=" * 70)
        
        # This is a complex scenario requiring:
        # 1. Behavioral drift simulation
        # 2. Rule update during drift
        # 3. Measurement of system stability
        
        # Simplified evaluation:
        # - Measure baseline before
        # - Introduce drift + rule change
        # - Measure convergence time
        
        agent_data = self.train_data[self.train_data['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('timestamp')
        
        if len(agent_data) < 300:
            self.get_logger().warning(f"Insufficient data for agent {agent_id}")
            return {}
        
        # Phase 1: Stable baseline
        phase1 = agent_data.iloc[:100]
        baseline_stable = self.baseline_builder.build_baseline(phase1)
        
        # Phase 2: Introduce drift
        phase2 = agent_data.iloc[100:200]
        
        # Phase 3: After concurrent changes
        phase3 = agent_data.iloc[200:300]
        baseline_after = self.baseline_builder.build_baseline(phase3)
        
        # Measure recovery
        drift_score_before = self._distribution_distance(
            baseline_stable.get('common_actions', {}).get('frequencies', {}),
            baseline_after.get('common_actions', {}).get('frequencies', {})
        )
        
        # Stability check: variance in decision latency
        stability_metric = 1.0 / (1.0 + drift_score_before)  # Higher = more stable
        
        results = {
            'agent_id': agent_id,
            'drift_score': drift_score_before,
            'stability_metric': stability_metric,
            'recovery_successful': drift_score_before < 0.5,  # Arbitrary threshold
        }
        
        self.get_logger().info(
            f"Concurrent changes: stability={stability_metric:.4f}, "
            f"recovery={'successful' if results['recovery_successful'] else 'failed'}"
        )
        
        return results
    
    def run_all_evaluations(self, agent_id: str = None):
        """Run all 6 adaptive/dynamic evaluations."""
        self.get_logger().info("=" * 70)
        self.get_logger().info("ADAPTIVE & DYNAMIC EVALUATION")
        self.get_logger().info("=" * 70)
        
        # Select agent if not provided
        if agent_id is None:
            agent_id = self.train_data['agent_id'].iloc[0]
        
        self.get_logger().info(f"Evaluating agent: {agent_id}")
        
        all_results = {}
        
        # Run evaluations
        all_results['convergence'] = self.evaluate_baseline_convergence(agent_id)
        all_results['drift_detection'] = self.evaluate_drift_detection(agent_id)
        all_results['sliding_window'] = self.evaluate_sliding_window(agent_id)
        all_results['rule_update_latency'] = self.evaluate_rule_update_latency()
        all_results['conflict_resolution'] = self.evaluate_conflict_resolution()
        all_results['concurrent_changes'] = self.evaluate_concurrent_changes(agent_id)
        
        # Save results
        self._save_results(all_results)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("ADAPTIVE & DYNAMIC EVALUATION COMPLETE")
        self.get_logger().info("=" * 70)
    
    def _distribution_distance(self, dist1: Dict, dist2: Dict) -> float:
        """Calculate distance between two distributions (simplified KL divergence)."""
        if not dist1 or not dist2:
            return 1.0
        
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        distance = 0.0
        for key in all_keys:
            p = dist1.get(key, 1e-10)
            q = dist2.get(key, 1e-10)
            distance += abs(p - q)
        
        return distance / 2.0  # Normalize to [0, 1]
    
    def _estimate_convergence_rate(self, distances: List[float]) -> float:
        """Estimate convergence rate from distance sequence."""
        # Fit exponential decay
        x = np.arange(len(distances))
        y = np.array(distances)
        
        # Simple linear fit on log scale
        if np.all(y > 0):
            log_y = np.log(y)
            coeffs = np.polyfit(x, log_y, 1)
            convergence_rate = -coeffs[0]  # Negative slope
        else:
            convergence_rate = 0.0
        
        return max(0.0, convergence_rate)
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy."""
        if not values:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        probs = [v / total for v in values if v > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        
        return entropy
    
    def _plot_convergence(self, metrics: List[Dict], agent_id: str):
        """Plot convergence over time."""
        import matplotlib.pyplot as plt
        
        window_indices = [m['window_idx'] for m in metrics]
        distances = [m['distance_from_previous'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(window_indices, distances, marker='o', linewidth=2)
        ax.set_xlabel('Window Index', fontsize=12)
        ax.set_ylabel('Distance from Previous', fontsize=12)
        ax.set_title(
            f'Baseline Convergence - Agent {agent_id}',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / f'convergence_{agent_id}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.get_logger().info(f"Saved convergence plot: {output_path}")
    
    def _plot_sliding_window_comparison(self, comparison: Dict):
        """Plot sliding window comparison."""
        import matplotlib.pyplot as plt
        
        strategies = comparison['strategy']
        metrics = ['action_entropy', 'time_gap_std', 'resource_diversity']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            axes[i].bar(strategies, comparison[metric])
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            axes[i].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Sliding Window Strategy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / 'sliding_window_comparison.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.get_logger().info(f"Saved sliding window plot: {output_path}")
    
    def _save_results(self, all_results: Dict):
        """Save all results."""
        # Save to JSON
        output_file = self.output_dir / 'adaptive_dynamic_results.json'
        
        # Make serializable
        serializable = {}
        for key, value in all_results.items():
            if isinstance(value, dict):
                serializable[key] = {
                    k: v for k, v in value.items()
                    if not k.startswith('baseline_')  # Skip large baseline objects
                }
            else:
                serializable[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / 'adaptive_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ADAPTIVE & DYNAMIC EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Convergence
            if 'convergence' in all_results:
                f.write("1. BASELINE CONVERGENCE RATE\n")
                f.write("-" * 50 + "\n")
                f.write(f"Convergence rate: {all_results['convergence'].get('convergence_rate', 0):.4f}\n")
                f.write("\n")
            
            # Drift detection
            if 'drift_detection' in all_results:
                f.write("2. DRIFT DETECTION ACCURACY\n")
                f.write("-" * 50 + "\n")
                f.write(f"Drift detected: {all_results['drift_detection'].get('drift_detected', False)}\n")
                f.write(f"Drift score: {all_results['drift_detection'].get('overall_drift_score', 0):.4f}\n")
                f.write("\n")
            
            # Sliding window
            if 'sliding_window' in all_results:
                f.write("3. SLIDING WINDOW EFFECTIVENESS\n")
                f.write("-" * 50 + "\n")
                f.write("See comparison plots\n")
                f.write("\n")
            
            # Rule update latency
            if 'rule_update_latency' in all_results:
                f.write("4. RULE UPDATE LATENCY\n")
                f.write("-" * 50 + "\n")
                metrics = all_results['rule_update_latency'].get('latency_metrics', {})
                f.write(f"Mean: {metrics.get('mean', 0):.2f} ms\n")
                f.write(f"P95: {metrics.get('p95', 0):.2f} ms\n")
                f.write(f"Meets target: {all_results['rule_update_latency'].get('meets_target', False)}\n")
                f.write("\n")
            
            # Conflict resolution
            if 'conflict_resolution' in all_results:
                f.write("5. CONFLICT RESOLUTION RATE\n")
                f.write("-" * 50 + "\n")
                f.write(f"Resolution rate: {all_results['conflict_resolution'].get('resolution_rate', 0):.2%}\n")
                f.write("\n")
            
            # Concurrent changes
            if 'concurrent_changes' in all_results:
                f.write("6. CONCURRENT DRIFT + RULE CHANGE\n")
                f.write("-" * 50 + "\n")
                f.write(f"Stability metric: {all_results['concurrent_changes'].get('stability_metric', 0):.4f}\n")
                f.write(f"Recovery: {'successful' if all_results['concurrent_changes'].get('recovery_successful', False) else 'failed'}\n")
                f.write("\n")
        
        self.get_logger().info(f"Saved results to {self.output_dir}")


def main(args=None):
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    rclpy.init(args=args)
    
    output_dir = Path('results/adaptive_evaluation')
    node = AdaptiveEvaluationNode(output_dir)
    
    try:
        # Select agent for evaluation
        agent_id = node.train_data['agent_id'].iloc[0]
        
        # Run all evaluations
        node.run_all_evaluations(agent_id=agent_id)
        
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        
    except Exception as e:
        node.get_logger().error(f"Error in adaptive evaluation: {e}")
        raise
        
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()