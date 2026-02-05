#!/usr/bin/env python3
"""
BBAC ICS Framework - Model Training Script

Trains all ML models:
1. Isolation Forest (anomaly detection)
2. LSTM Sequence Model (action prediction)
3. Meta-Classifier (layer fusion)

Saves trained models to trained_models/ directory.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data.loader import DatasetLoader
from ..models.statistical import IsolationForestModel
from ..models.lstm import LSTMSequenceModel, BidirectionalLSTMModel
from ..models.fusion import MetaClassifier, EnsembleFusion
from ..core.modeling import BaselineBuilder
from ..core.analysis import AnalysisLayer
from ..util.config_loader import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and saves all BBAC models."""
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        train_isolation_forest: bool = True,
        train_lstm: bool = True,
        train_meta_classifier: bool = True,
    ):
        """
        Initialize model trainer.
        
        Args:
            data_dir: Directory containing dataset
            output_dir: Directory to save trained models
            train_isolation_forest: Whether to train Isolation Forest
            train_lstm: Whether to train LSTM
            train_meta_classifier: Whether to train meta-classifier
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_isolation_forest = train_isolation_forest
        self.train_lstm = train_lstm
        self.train_meta_classifier = train_meta_classifier
        
        # Load dataset
        logger.info("=" * 70)
        logger.info("LOADING DATASET")
        logger.info("=" * 70)
        
        self.loader = DatasetLoader(self.data_dir)
        self.loader.load_all()
        
        logger.info(f"Train data: {len(self.loader.train_data)} samples")
        logger.info(f"Validation data: {len(self.loader.validation_data)} samples")
        logger.info(f"Test data: {len(self.loader.test_data)} samples")
    
    def train_all(self):
        """Train all models."""
        logger.info("=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)
        
        if self.train_isolation_forest:
            self._train_isolation_forest()
        
        if self.train_lstm:
            self._train_lstm()
        
        if self.train_meta_classifier:
            self._train_meta_classifier()
        
        logger.info("=" * 70)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 70)
    
    def _train_isolation_forest(self):
        """Train Isolation Forest for anomaly detection."""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING ISOLATION FOREST")
        logger.info("=" * 70)
        
        # Extract features from training data
        logger.info("Extracting features...")
        features = self._extract_features_for_isolation_forest(
            self.loader.train_data
        )
        
        logger.info(f"Extracted features: shape={features.shape}")
        
        # Train model
        contamination = config.ml_params.get('statistical', {}).get('contamination', 0.1)
        
        model = IsolationForestModel(
            contamination=contamination,
            n_estimators=100,
            random_state=42
        )
        
        model.fit(features)
        
        # Validate
        logger.info("Validating on validation set...")
        val_features = self._extract_features_for_isolation_forest(
            self.loader.validation_data
        )
        
        predictions = model.predict(val_features)
        
        # Calculate metrics
        mean_score = np.mean(predictions)
        std_score = np.std(predictions)
        
        logger.info(
            f"Validation scores: mean={mean_score:.4f}, std={std_score:.4f}"
        )
        
        # Save model
        output_path = self.output_dir / 'isolation_forest.pkl'
        model.save_model(str(output_path))
        
        logger.info(f"✓ Isolation Forest saved to {output_path}")
    
    def _extract_features_for_isolation_forest(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Extract numerical features for Isolation Forest."""
        features_list = []
        
        for idx, row in data.iterrows():
            # Extract temporal features
            timestamp = pd.to_datetime(row['timestamp'])
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            
            # Action encoding (simple)
            action_map = {'read': 0, 'write': 1, 'execute': 2, 'delete': 3, 'override': 4}
            action_encoded = action_map.get(row['action'], 0)
            
            # Resource type encoding
            resource_type_map = {
                'database': 0, 'actuator': 1, 'sensor': 2,
                'conveyor': 3, 'camera': 4, 'admin_panel': 5
            }
            resource_type_encoded = resource_type_map.get(row['resource_type'], 0)
            
            # Agent type encoding
            agent_type_encoded = 1 if row['agent_type'] == 'robot' else 0
            
            # Boolean features
            human_present = 1 if row['human_present'] else 0
            emergency = 1 if row['emergency_flag'] else 0
            
            # Location encoding (hash)
            location_hash = hash(row['location']) % 10
            
            feature_vector = [
                hour,
                day_of_week,
                action_encoded,
                resource_type_encoded,
                agent_type_encoded,
                human_present,
                emergency,
                location_hash,
            ]
            
            features_list.append(feature_vector)
        
        return np.array(features_list, dtype=np.float32)
    
    def _train_lstm(self):
        """Train LSTM sequence model."""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING LSTM SEQUENCE MODEL")
        logger.info("=" * 70)
        
        # Extract action sequences
        logger.info("Extracting action sequences...")
        sequences = self._extract_action_sequences(self.loader.train_data)
        
        logger.info(f"Extracted {len(sequences)} sequences")
        
        # Train LSTM
        model = LSTMSequenceModel(
            sequence_length=5,
            embedding_dim=16,
            lstm_units=32,
            dropout=0.2,
            batch_size=32,
            epochs=50,
        )
        
        model.fit(
            sequences,
            validation_split=0.2,
            verbose=1
        )
        
        # Validate
        logger.info("Validating on test sequences...")
        test_sequences = self._extract_action_sequences(self.loader.test_data)
        
        # Test prediction accuracy
        correct = 0
        total = 0
        
        for seq in test_sequences[:100]:  # Sample 100 sequences
            if len(seq) > model.sequence_length:
                input_seq = seq[:-1]
                target_action = seq[-1]
                
                predictions = model.predict_next_action(input_seq, top_k=3)
                predicted_actions = [p[0] for p in predictions]
                
                if target_action in predicted_actions:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Top-3 prediction accuracy: {accuracy:.2%}")
        
        # Save model
        output_path = self.output_dir / 'lstm_sequence_model'
        model.save_model(str(output_path))
        
        logger.info(f"✓ LSTM model saved to {output_path}.h5 and {output_path}.pkl")
        
        # Train Bidirectional variant
        logger.info("\nTraining Bidirectional LSTM...")
        
        bilstm_model = BidirectionalLSTMModel(
            sequence_length=5,
            embedding_dim=16,
            lstm_units=32,
            dropout=0.2,
            batch_size=32,
            epochs=50,
        )
        
        bilstm_model.fit(
            sequences,
            validation_split=0.2,
            verbose=1
        )
        
        # Save Bidirectional model
        bilstm_path = self.output_dir / 'bilstm_sequence_model'
        bilstm_model.save_model(str(bilstm_path))
        
        logger.info(f"✓ Bidirectional LSTM saved to {bilstm_path}.h5")
    
    def _extract_action_sequences(
        self,
        data: pd.DataFrame
    ) -> List[List[str]]:
        """Extract action sequences grouped by agent and session."""
        # Group by agent_id and session_id
        sequences = []
        
        for (agent_id, session_id), group in data.groupby(['user_id', 'session_id']):
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Extract actions
            actions = group['action'].tolist()
            
            if len(actions) >= 3:  # Minimum sequence length
                sequences.append(actions)
        
        return sequences
    
    def _train_meta_classifier(self):
        """Train meta-classifier for layer fusion."""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING META-CLASSIFIER")
        logger.info("=" * 70)
        
        # Generate layer decisions for training data
        logger.info("Generating layer decisions...")
        
        layer_results_list, ground_truth = self._generate_layer_decisions(
            self.loader.train_data
        )
        
        logger.info(f"Generated {len(layer_results_list)} layer decision samples")
        
        # Train Logistic Regression
        logger.info("\nTraining Logistic Regression meta-classifier...")
        
        logreg_model = MetaClassifier(
            model_type='logistic_regression',
            max_iter=1000,
            random_state=42
        )
        
        logreg_model.fit(layer_results_list, ground_truth)
        
        # Save
        logreg_path = self.output_dir / 'meta_logreg.pkl'
        logreg_model.save_model(str(logreg_path))
        logger.info(f"✓ LogReg meta-classifier saved to {logreg_path}")
        
        # Get learned weights
        weights = logreg_model.get_learned_weights()
        logger.info(f"Learned weights: {weights}")
        
        # Train XGBoost (if available)
        try:
            logger.info("\nTraining XGBoost meta-classifier...")
            
            xgb_model = MetaClassifier(
                model_type='xgboost',
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
            
            xgb_model.fit(layer_results_list, ground_truth)
            
            # Save
            xgb_path = self.output_dir / 'meta_xgboost.pkl'
            xgb_model.save_model(str(xgb_path))
            logger.info(f"✓ XGBoost meta-classifier saved to {xgb_path}")
            
            # Get learned weights
            xgb_weights = xgb_model.get_learned_weights()
            logger.info(f"XGBoost learned weights: {xgb_weights}")
            
            # Train ensemble
            logger.info("\nTraining ensemble meta-classifier...")
            
            ensemble = EnsembleFusion([logreg_model, xgb_model])
            
            # Validate ensemble
            val_layer_results, val_ground_truth = self._generate_layer_decisions(
                self.loader.validation_data.head(500)  # Sample for speed
            )
            
            correct = 0
            for lr, gt in zip(val_layer_results, val_ground_truth):
                pred, conf = ensemble.predict(lr, voting='soft')
                if pred == gt:
                    correct += 1
            
            ensemble_acc = correct / len(val_ground_truth)
            logger.info(f"Ensemble validation accuracy: {ensemble_acc:.2%}")
            
            # Get averaged weights
            avg_weights = ensemble.get_averaged_weights()
            logger.info(f"Ensemble averaged weights: {avg_weights}")
            
        except ImportError:
            logger.warning("XGBoost not available, skipping XGBoost training")
    
    def _generate_layer_decisions(
        self,
        data: pd.DataFrame
    ) -> tuple:
        """
        Generate synthetic layer decisions for training meta-classifier.
        
        Args:
            data: DataFrame with requests
            
        Returns:
            Tuple of (layer_results_list, ground_truth)
        """
        from ..core.ingestion import IngestionLayer
        from ..core.modeling import ModelingLayer
        
        # Initialize layers
        ingestion = IngestionLayer(self.loader)
        modeling = ModelingLayer()
        analysis = AnalysisLayer()
        
        # Build baselines first
        logger.info("Building agent baselines...")
        baseline_builder = BaselineBuilder()
        
        for agent_id in data['user_id'].unique():
            agent_data = data[data['user_id'] == agent_id]
            if len(agent_data) >= 10:
                baseline = baseline_builder.build_baseline(agent_data)
                modeling.profile_manager.profiles[agent_id] = {
                    'baseline': baseline,
                    'agent_type': agent_data.iloc[0]['agent_type']
                }
        
        layer_results_list = []
        ground_truth = []
        
        # Sample data for speed
        sample_data = data.head(min(1000, len(data)))
        
        for idx, row in sample_data.iterrows():
            try:
                # Convert to AccessRequest
                request = self.loader.to_access_request(row)
                
                # Get profile
                profile = modeling.get_agent_profile(request.agent_id)
                baseline = profile.get('baseline') if profile else None
                
                # Prepare features
                features = modeling.prepare_features(
                    {
                        'agent_id': request.agent_id,
                        'action': request.action.value,
                        'resource': request.resource,
                        'location': request.location,
                    },
                    request.agent_id
                )
                
                # Get layer results
                layer_results = analysis.analyze_request(
                    request,
                    baseline,
                    features,
                    enable_stat=True,
                    enable_sequence=False,  # Skip sequence for speed
                    enable_policy=True,
                )
                
                # Convert LayerDecision to dict
                results_dict = {}
                for layer_name, layer_decision in layer_results.items():
                    results_dict[layer_name] = {
                        'decision': layer_decision.decision.value,
                        'confidence': layer_decision.confidence,
                        'latency_ms': layer_decision.latency_ms,
                    }
                
                layer_results_list.append(results_dict)
                
                # Ground truth (use anomaly label if available, otherwise grant)
                if 'is_anomaly' in row:
                    gt = 'deny' if row['is_anomaly'] else 'grant'
                elif 'expected_decision' in row:
                    gt = row['expected_decision']
                else:
                    # Heuristic: emergency or high attempt count → deny
                    if row.get('emergency_flag', False) or row.get('attempt_count', 0) > 3:
                        gt = 'deny'
                    else:
                        gt = 'grant'
                
                ground_truth.append(gt)
                
            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue
        
        return layer_results_list, ground_truth
    
    def generate_training_report(self):
        """Generate training summary report."""
        report_path = self.output_dir / 'training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("BBAC FRAMEWORK - MODEL TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DATASET SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Training samples: {len(self.loader.train_data)}\n")
            f.write(f"Validation samples: {len(self.loader.validation_data)}\n")
            f.write(f"Test samples: {len(self.loader.test_data)}\n")
            f.write("\n")
            
            f.write("TRAINED MODELS\n")
            f.write("-" * 50 + "\n")
            
            if (self.output_dir / 'isolation_forest.pkl').exists():
                f.write("✓ Isolation Forest (anomaly detection)\n")
            
            if (self.output_dir / 'lstm_sequence_model.h5').exists():
                f.write("✓ LSTM Sequence Model\n")
            
            if (self.output_dir / 'bilstm_sequence_model.h5').exists():
                f.write("✓ Bidirectional LSTM\n")
            
            if (self.output_dir / 'meta_logreg.pkl').exists():
                f.write("✓ Meta-Classifier (Logistic Regression)\n")
            
            if (self.output_dir / 'meta_xgboost.pkl').exists():
                f.write("✓ Meta-Classifier (XGBoost)\n")
            
            f.write("\n")
            f.write("MODEL FILES\n")
            f.write("-" * 50 + "\n")
            
            for file in sorted(self.output_dir.iterdir()):
                f.write(f"  {file.name}\n")
        
        logger.info(f"\n✓ Training report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train BBAC ML models'
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
        default='trained_models',
        help='Output directory for trained models'
    )
    
    parser.add_argument(
        '--skip-isolation-forest',
        action='store_true',
        help='Skip Isolation Forest training'
    )
    
    parser.add_argument(
        '--skip-lstm',
        action='store_true',
        help='Skip LSTM training'
    )
    
    parser.add_argument(
        '--skip-meta-classifier',
        action='store_true',
        help='Skip meta-classifier training'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_isolation_forest=not args.skip_isolation_forest,
        train_lstm=not args.skip_lstm,
        train_meta_classifier=not args.skip_meta_classifier,
    )
    
    # Train all models
    trainer.train_all()
    
    # Generate report
    trainer.generate_training_report()
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()