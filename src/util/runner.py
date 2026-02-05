#!/usr/bin/env python3
"""
BBAC ICS Framework - Offline Pipeline Runner

Runs BBAC pipeline without ROS2 for:
- Quick testing
- Debugging
- Batch processing
- Performance benchmarking

Usage:
    python util/runner.py --data-dir data/100k --output-dir results/offline --max-samples 1000
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from data.loader import DatasetLoader
from ..core.analysis import AnalysisLayer
from ..core.decision import DecisionLayer
from ..core.fusion import FusionLayer
from ..core.ingestion import IngestionLayer
from ..core.learning import ContinuousLearningLayer
from ..core.modeling import ModelingLayer
from ..util.evaluator import MetricsEvaluator
from ..util.config_loader import config
from ..util.visualizer import Visualizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class BBACRunner:
    """
    Offline BBAC pipeline runner.
    
    Processes requests through all 5 layers without ROS2.
    """
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        use_lstm: bool = False,
        use_isolation_forest: bool = False,
        use_meta_classifier: bool = False,
        models_dir: str = 'trained_models'
    ):
        """
        Initialize BBAC runner.
        
        Args:
            data_dir: Dataset directory
            output_dir: Output directory for results
            use_lstm: Use LSTM for sequence analysis
            use_isolation_forest: Use Isolation Forest for anomaly detection
            use_meta_classifier: Use meta-classifier for fusion
            models_dir: Directory with trained models
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = models_dir
        
        logger.info("=" * 70)
        logger.info("BBAC OFFLINE RUNNER - INITIALIZATION")
        logger.info("=" * 70)
        
        # Load dataset
        logger.info("Loading dataset...")
        self.loader = DatasetLoader(self.data_dir)
        self.loader.load_all()
        
        logger.info(f"Loaded {len(self.loader.test_data)} test samples")
        
        # Initialize all 5 layers
        logger.info("Initializing BBAC layers...")
        
        # Layer 1: Ingestion
        self.ingestion_layer = IngestionLayer(self.loader)
        logger.info("✓ Layer 1 (Ingestion)")
        
        # Layer 2: Modeling
        self.modeling_layer = ModelingLayer()
        logger.info("✓ Layer 2 (Modeling)")
        
        # Layer 3: Analysis
        self.analysis_layer = AnalysisLayer(
            use_lstm=use_lstm,
            use_isolation_forest=use_isolation_forest,
            models_dir=models_dir
        )
        logger.info(f"✓ Layer 3 (Analysis) - LSTM={use_lstm}, IForest={use_isolation_forest}")
        
        # Build sequence models
        if not use_lstm:
            self._build_sequence_models()
        
        # Layer 4: Fusion + Decision
        self.fusion_layer = FusionLayer(
            use_meta_classifier=use_meta_classifier,
            models_dir=models_dir
        )
        logger.info(f"✓ Layer 4a (Fusion) - MetaClassifier={use_meta_classifier}")
        
        self.decision_layer = DecisionLayer(self.output_dir / 'logs')
        logger.info("✓ Layer 4b (Decision)")
        
        # Layer 5: Learning
        self.learning_layer = ContinuousLearningLayer(
            profile_manager=self.modeling_layer.profile_manager,
            log_dir=self.output_dir / 'logs'
        )
        logger.info("✓ Layer 5 (Learning)")
        
        # Evaluator and visualizer
        self.evaluator = MetricsEvaluator()
        self.visualizer = Visualizer(self.output_dir / 'figures')
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'grants': 0,
            'denials': 0,
            'approvals': 0,
            'errors': 0,
            'latencies': [],
        }
        
        logger.info("=" * 70)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("=" * 70)
    
    def _build_sequence_models(self):
        """Build Markov chain sequence models from training data."""
        logger.info("Building Markov chain models...")
        
        train_data = self.loader.train_data
        
        if train_data is None or train_data.empty:
            logger.warning("No training data for sequence models")
            return
        
        # Group by agent
        agents = train_data['user_id'].unique()
        
        for agent_id in agents:
            agent_data = train_data[train_data['user_id'] == agent_id]
            
            # Build transition matrix
            self.analysis_layer.sequence_engine.build_transition_matrix(
                agent_id,
                agent_data
            )
        
        logger.info(f"Built sequence models for {len(agents)} agents")
    
    def process_request(self, request) -> Dict:
        """
        Process single request through all 5 layers.
        
        Args:
            request: AccessRequest object
            
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        
        try:
            # Layer 1: Ingestion
            processed_request = self.ingestion_layer.process_request(request)
            
            if processed_request is None:
                # Authentication failed
                return {
                    'request_id': request.request_id,
                    'decision': 'deny',
                    'confidence': 1.0,
                    'latency_ms': (time.time() - start_time) * 1000,
                    'reason': 'authentication_failed',
                    'error': False,
                }
            
            # Layer 2: Modeling
            profile = self.modeling_layer.get_agent_profile(request.agent_id)
            baseline = profile.get('baseline') if profile else None
            
            features = self.modeling_layer.prepare_features(
                {
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
                request.agent_id
            )
            
            # Layer 3: Analysis
            layer_results = self.analysis_layer.analyze_request(
                processed_request,
                baseline,
                features,
                enable_stat=True,
                enable_sequence=True,
                enable_policy=True,
            )
            
            # Layer 4a: Fusion
            fused_result = self.fusion_layer.fuse(layer_results)
            
            # Layer 4b: Decision
            decision = self.decision_layer.make_decision(
                processed_request,
                fused_result
            )
            
            # Layer 5: Learning
            fused_score = fused_result.get('confidence', 0.5)
            self.learning_layer.process_decision(
                processed_request,
                decision,
                fused_score
            )
            
            # Update statistics
            self.stats['total_processed'] += 1
            
            if decision.decision.value == 'grant':
                self.stats['grants'] += 1
            elif decision.decision.value == 'deny':
                self.stats['denials'] += 1
            elif decision.decision.value == 'require_approval':
                self.stats['approvals'] += 1
            
            self.stats['latencies'].append(decision.latency_ms)
            
            return {
                'request_id': request.request_id,
                'decision': decision.decision.value,
                'confidence': decision.confidence,
                'latency_ms': decision.latency_ms,
                'reason': decision.reason,
                'layer_results': fused_result.get('layer_results', {}),
                'error': False,
            }
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            self.stats['errors'] += 1
            
            return {
                'request_id': request.request_id,
                'decision': 'deny',
                'confidence': 0.0,
                'latency_ms': (time.time() - start_time) * 1000,
                'reason': f'processing_error: {str(e)}',
                'error': True,
            }
    
    def run(
        self,
        max_samples: Optional[int] = None,
        dataset: str = 'test'
    ) -> Dict:
        """
        Run pipeline on dataset.
        
        Args:
            max_samples: Maximum number of samples to process
            dataset: Dataset to use ('train', 'validation', 'test')
            
        Returns:
            Results dictionary
        """
        logger.info("=" * 70)
        logger.info("RUNNING BBAC PIPELINE")
        logger.info("=" * 70)
        
        # Get dataset
        if dataset == 'train':
            data = self.loader.train_data
        elif dataset == 'validation':
            data = self.loader.validation_data
        else:
            data = self.loader.test_data
        
        if data is None or data.empty:
            raise ValueError(f"No {dataset} data available")
        
        # Limit samples
        if max_samples:
            data = data.head(max_samples)
        
        logger.info(f"Processing {len(data)} samples from {dataset} set...")
        
        # Process all requests
        results = []
        predictions = []
        ground_truth = []
        confidences = []
        
        start_time = time.time()
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing"):
            try:
                # Convert to AccessRequest
                request = self.loader.to_access_request(row)
                
                # Process request
                result = self.process_request(request)
                results.append(result)
                
                # Collect for evaluation
                predictions.append(result['decision'])
                confidences.append(result['confidence'])
                
                # Ground truth
                if 'expected_decision' in row:
                    gt = row['expected_decision']
                elif 'is_anomaly' in row:
                    gt = 'deny' if row['is_anomaly'] else 'grant'
                else:
                    gt = 'grant'  # Default
                
                ground_truth.append(gt)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Evaluate
        logger.info("\nEvaluating results...")
        
        classification_metrics = self.evaluator.evaluate_classification(
            ground_truth,
            predictions,
            confidences
        )
        
        performance_metrics = self.evaluator.evaluate_performance(
            self.stats['latencies'],
            total_time
        )
        
        # Save results
        self._save_results(results, classification_metrics, performance_metrics)
        
        # Generate visualizations
        self._generate_visualizations(
            classification_metrics,
            ground_truth,
            predictions
        )
        
        # Print summary
        self._print_summary(classification_metrics, performance_metrics)
        
        return {
            'results': results,
            'classification': classification_metrics,
            'performance': performance_metrics,
            'statistics': self.stats,
        }
    
    def _save_results(
        self,
        results: List[Dict],
        classification_metrics,
        performance_metrics
    ):
        """Save results to files."""
        # Save individual results
        results_file = self.output_dir / 'results.jsonl'
        
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Saved results to {results_file}")
        
        # Save metrics
        metrics_file = self.output_dir / 'metrics.json'
        
        with open(metrics_file, 'w') as f:
            json.dump({
                'classification': classification_metrics.to_dict(),
                'performance': performance_metrics.to_dict(),
                'statistics': self.stats,
            }, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")
    
    def _generate_visualizations(
        self,
        classification_metrics,
        ground_truth: List[str],
        predictions: List[str]
    ):
        """Generate visualization plots."""
        logger.info("Generating visualizations...")
        
        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            classification_metrics,
            title='BBAC Offline Runner - Confusion Matrix',
            filename='confusion_matrix.png'
        )
        
        # Latency distribution
        self.visualizer.plot_latency_distribution(
            {'BBAC': self.stats['latencies']},
            title='BBAC Latency Distribution',
            filename='latency_distribution.png'
        )
        
        logger.info(f"Saved visualizations to {self.output_dir / 'figures'}")
    
    def _print_summary(
        self,
        classification_metrics,
        performance_metrics
    ):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("BBAC OFFLINE RUNNER - SUMMARY")
        print("=" * 70)
        
        print(f"\nProcessed: {self.stats['total_processed']} requests")
        print(f"  Grants: {self.stats['grants']} ({self.stats['grants']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Denials: {self.stats['denials']} ({self.stats['denials']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Approvals: {self.stats['approvals']} ({self.stats['approvals']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Errors: {self.stats['errors']}")
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {classification_metrics.accuracy:.4f}")
        print(f"  Precision: {classification_metrics.precision:.4f}")
        print(f"  Recall:    {classification_metrics.recall:.4f}")
        print(f"  F1 Score:  {classification_metrics.f1:.4f}")
        print(f"  ROC-AUC:   {classification_metrics.roc_auc:.4f}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Mean Latency: {performance_metrics.latency.mean:.2f} ms")
        print(f"  P95 Latency:  {performance_metrics.latency.p95:.2f} ms")
        print(f"  P99 Latency:  {performance_metrics.latency.p99:.2f} ms")
        print(f"  Throughput:   {performance_metrics.throughput:.2f} req/s")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='BBAC Offline Pipeline Runner'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/100k',
        help='Dataset directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/offline',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--use-lstm',
        action='store_true',
        help='Use LSTM for sequence analysis'
    )
    
    parser.add_argument(
        '--use-isolation-forest',
        action='store_true',
        help='Use Isolation Forest for anomaly detection'
    )
    
    parser.add_argument(
        '--use-meta-classifier',
        action='store_true',
        help='Use meta-classifier for fusion'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='trained_models',
        help='Directory with trained models'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = BBACRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_lstm=args.use_lstm,
        use_isolation_forest=args.use_isolation_forest,
        use_meta_classifier=args.use_meta_classifier,
        models_dir=args.models_dir
    )
    
    # Run pipeline
    runner.run(
        max_samples=args.max_samples,
        dataset=args.dataset
    )


if __name__ == '__main__':
    main()