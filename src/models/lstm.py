#!/usr/bin/env python3
"""
BBAC ICS Framework - LSTM Sequence Model

LSTM-based sequence analysis for action prediction:
- Learns temporal patterns from action sequences
- Predicts next action based on history
- Detects anomalies when actions deviate from expected patterns

Upgrade from Markov chains with better long-term memory.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LSTMSequenceModel:
    """
    LSTM model for sequence analysis and anomaly detection.
    
    Learns action sequences and predicts next actions.
    """
    
    def __init__(
        self,
        sequence_length: int = 5,
        embedding_dim: int = 16,
        lstm_units: int = 32,
        dropout: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of past actions to consider
            embedding_dim: Embedding dimension for actions
            lstm_units: Number of LSTM units
            dropout: Dropout rate for regularization
            batch_size: Training batch size
            epochs: Training epochs
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Install with: "
                "pip install 'tensorflow>=2.13.0,<2.16.0'"
            )
        
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Action vocabulary (will be built during fit)
        self.action_vocab = {}  # action_name -> index
        self.reverse_vocab = {}  # index -> action_name
        self.vocab_size = 0
        
        # Model
        self.model = None
        self.is_fitted = False
        
        logger.info(
            f"LSTMSequenceModel initialized: seq_len={sequence_length}, "
            f"lstm_units={lstm_units}"
        )
    
    def _build_vocabulary(self, action_sequences: List[List[str]]):
        """Build action vocabulary from sequences."""
        unique_actions = set()
        for seq in action_sequences:
            unique_actions.update(seq)
        
        # Create mappings
        self.action_vocab = {
            action: idx for idx, action in enumerate(sorted(unique_actions))
        }
        self.reverse_vocab = {
            idx: action for action, idx in self.action_vocab.items()
        }
        self.vocab_size = len(self.action_vocab)
        
        logger.info(f"Vocabulary built: {self.vocab_size} unique actions")
        logger.debug(f"Actions: {list(self.action_vocab.keys())}")
    
    def _build_model(self):
        """Build LSTM architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='action_embedding'
            ),
            
            # LSTM layer
            layers.LSTM(
                self.lstm_units,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                name='lstm'
            ),
            
            # Dense layers
            layers.Dense(
                self.lstm_units // 2,
                activation='relu',
                name='dense'
            ),
            layers.Dropout(self.dropout),
            
            # Output layer (predict next action)
            layers.Dense(
                self.vocab_size,
                activation='softmax',
                name='output'
            )
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        logger.info("LSTM model architecture built")
        logger.debug(f"Model summary:\n{model.summary()}")
    
    def _prepare_sequences(
        self,
        action_sequences: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training sequences.
        
        Args:
            action_sequences: List of action sequences (each is list of action names)
            
        Returns:
            Tuple of (X, y) where:
                X: Input sequences (n_samples, sequence_length)
                y: Target actions (n_samples,)
        """
        X_list = []
        y_list = []
        
        for sequence in action_sequences:
            # Convert actions to indices
            indices = [
                self.action_vocab.get(action, 0)
                for action in sequence
            ]
            
            # Create sliding windows
            for i in range(len(indices) - self.sequence_length):
                X_list.append(indices[i:i + self.sequence_length])
                y_list.append(indices[i + self.sequence_length])
        
        X = np.array(X_list, dtype=np.int32)
        y = np.array(y_list, dtype=np.int32)
        
        logger.info(f"Prepared {len(X)} training sequences")
        
        return X, y
    
    def fit(
        self,
        action_sequences: List[List[str]],
        validation_split: float = 0.2,
        verbose: int = 1
    ):
        """
        Train LSTM model on action sequences.
        
        Args:
            action_sequences: List of action sequences
                Example: [
                    ['read', 'execute', 'read', 'write'],
                    ['read', 'read', 'execute', 'read'],
                    ...
                ]
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        # Build vocabulary
        self._build_vocabulary(action_sequences)
        
        # Build model
        self._build_model()
        
        # Prepare sequences
        X, y = self._prepare_sequences(action_sequences)
        
        if len(X) == 0:
            raise ValueError("No training sequences generated")
        
        # Train model
        logger.info(f"Training LSTM for {self.epochs} epochs...")
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=verbose
            )
        ]
        
        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        # Log final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        logger.info(
            f"Training complete: loss={final_loss:.4f}, acc={final_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
    
    def predict_next_action(
        self,
        action_sequence: List[str],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict next action(s) given a sequence.
        
        Args:
            action_sequence: Recent action sequence
            top_k: Number of top predictions to return
            
        Returns:
            List of (action, probability) tuples, sorted by probability
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Take last sequence_length actions
        sequence = action_sequence[-self.sequence_length:]
        
        # Pad if too short
        if len(sequence) < self.sequence_length:
            sequence = ['read'] * (self.sequence_length - len(sequence)) + sequence
        
        # Convert to indices
        indices = [
            self.action_vocab.get(action, 0)
            for action in sequence
        ]
        
        # Predict
        X = np.array([indices], dtype=np.int32)
        probs = self.model.predict(X, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        predictions = [
            (self.reverse_vocab[idx], float(probs[idx]))
            for idx in top_indices
        ]
        
        return predictions
    
    def calculate_anomaly_score(
        self,
        action_sequence: List[str],
        actual_next_action: str
    ) -> float:
        """
        Calculate anomaly score for an action sequence.
        
        Args:
            action_sequence: Historical action sequence
            actual_next_action: Actual next action that occurred
            
        Returns:
            Anomaly score in [0, 1] where:
                0 = highly anomalous (unexpected)
                1 = normal (expected)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Predict next actions
        predictions = self.predict_next_action(action_sequence, top_k=self.vocab_size)
        
        # Find probability of actual action
        for action, prob in predictions:
            if action == actual_next_action:
                # Convert probability to normalcy score
                # Higher probability → higher normalcy
                return float(prob)
        
        # Action not in vocabulary (very anomalous)
        return 0.0
    
    def analyze_sequence(
        self,
        action_sequence: List[str],
        current_action: str,
        threshold: float = 0.3
    ) -> Dict:
        """
        Analyze action sequence for anomalies.
        
        Args:
            action_sequence: Historical actions
            current_action: Current action to analyze
            threshold: Anomaly threshold
            
        Returns:
            Analysis result dictionary
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted:
            return {
                'decision': 'grant',
                'confidence': 0.5,
                'normalcy_score': 0.5,
                'latency_ms': (time.time() - start_time) * 1000,
                'reason': 'model_not_fitted',
            }
        
        # Calculate anomaly score
        normalcy_score = self.calculate_anomaly_score(
            action_sequence,
            current_action
        )
        
        # Get predictions for context
        predictions = self.predict_next_action(action_sequence, top_k=3)
        
        # Decision
        decision = 'grant' if normalcy_score >= threshold else 'deny'
        
        # Confidence (distance from threshold)
        if decision == 'grant':
            confidence = 0.5 + (normalcy_score - threshold) / (1.0 - threshold) * 0.5
        else:
            confidence = 0.5 + (threshold - normalcy_score) / threshold * 0.5
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'decision': decision,
            'confidence': confidence,
            'normalcy_score': float(normalcy_score),
            'latency_ms': float(latency_ms),
            'threshold': threshold,
            'expected_actions': predictions,
            'actual_action': current_action,
        }
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        import pickle
        
        filepath = Path(filepath)
        
        # Save Keras model
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)
        
        # Save vocabulary and config
        config_path = filepath.with_suffix('.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump({
                'action_vocab': self.action_vocab,
                'reverse_vocab': self.reverse_vocab,
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length,
                'embedding_dim': self.embedding_dim,
                'lstm_units': self.lstm_units,
                'dropout': self.dropout,
                'is_fitted': self.is_fitted,
            }, f)
        
        logger.info(f"Model saved to {model_path} and {config_path}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        import pickle
        from tensorflow import keras
        
        filepath = Path(filepath)
        
        # Load Keras model
        model_path = filepath.with_suffix('.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load vocabulary and config
        config_path = filepath.with_suffix('.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.action_vocab = config['action_vocab']
        self.reverse_vocab = config['reverse_vocab']
        self.vocab_size = config['vocab_size']
        self.sequence_length = config['sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config['lstm_units']
        self.dropout = config['dropout']
        self.is_fitted = config['is_fitted']
        
        logger.info(f"Model loaded from {model_path}")


class BidirectionalLSTMModel(LSTMSequenceModel):
    """
    Bidirectional LSTM variant for better context understanding.
    
    Uses both past and future context (during training) for learning.
    """
    
    def _build_model(self):
        """Build Bidirectional LSTM architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='action_embedding'
            ),
            
            # Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                ),
                name='bilstm'
            ),
            
            # Dense layers
            layers.Dense(
                self.lstm_units,
                activation='relu',
                name='dense'
            ),
            layers.Dropout(self.dropout),
            
            # Output layer
            layers.Dense(
                self.vocab_size,
                activation='softmax',
                name='output'
            )
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        logger.info("Bidirectional LSTM model architecture built")


__all__ = ['LSTMSequenceModel', 'BidirectionalLSTMModel']