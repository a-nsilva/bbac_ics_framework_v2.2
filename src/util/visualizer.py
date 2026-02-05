#!/usr/bin/env python3
"""
BBAC ICS Framework - Visualization

Generates publication-quality figures for research paper:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Latency distributions
- Comparison charts
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from .data_structures import ClassificationMetrics, LatencyMetrics


logger = logging.getLogger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class Visualizer:
    """
    Creates publication-quality visualizations for research paper.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures
        """
        if output_dir is None:
            output_dir = Path('results/figures')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visualizer initialized: output={self.output_dir}")
    
    def plot_confusion_matrix(
        self,
        metrics: ClassificationMetrics,
        title: str = 'Confusion Matrix',
        filename: str = 'confusion_matrix.png'
    ) -> Figure:
        """
        Plot confusion matrix.
        
        Args:
            metrics: Classification metrics with confusion matrix
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        # Create confusion matrix array
        cm = np.array([
            [metrics.tn, metrics.fp],
            [metrics.fn, metrics.tp]
        ])
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Deny', 'Grant'],
            yticklabels=['Deny', 'Grant'],
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add accuracy text
        accuracy = (metrics.tp + metrics.tn) / (metrics.tp + metrics.tn + metrics.fp + metrics.fn)
        ax.text(
            0.5, -0.15,
            f'Accuracy: {accuracy:.3f} | Precision: {metrics.precision:.3f} | '
            f'Recall: {metrics.recall:.3f} | F1: {metrics.f1:.3f}',
            ha='center',
            transform=ax.transAxes,
            fontsize=10
        )
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved confusion matrix: {output_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        metrics_dict: Dict[str, ClassificationMetrics],
        title: str = 'ROC Curve Comparison',
        filename: str = 'roc_curve.png'
    ) -> Figure:
        """
        Plot ROC curves for multiple methods.
        
        Args:
            metrics_dict: Dictionary mapping method names to metrics
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        
        for method_name, metrics in metrics_dict.items():
            if metrics.fpr and metrics.tpr:
                ax.plot(
                    metrics.fpr,
                    metrics.tpr,
                    label=f'{method_name} (AUC={metrics.roc_auc:.3f})',
                    linewidth=2
                )
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved ROC curve: {output_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        metrics_dict: Dict[str, ClassificationMetrics],
        title: str = 'Precision-Recall Curve',
        filename: str = 'pr_curve.png'
    ) -> Figure:
        """
        Plot Precision-Recall curves.
        
        Args:
            metrics_dict: Dictionary mapping method names to metrics
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        
        for method_name, metrics in metrics_dict.items():
            if metrics.precision_curve and metrics.recall_curve:
                ax.plot(
                    metrics.recall_curve,
                    metrics.precision_curve,
                    label=f'{method_name} (AP={metrics.avg_precision:.3f})',
                    linewidth=2
                )
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved PR curve: {output_path}")
        
        return fig
    
    def plot_latency_distribution(
        self,
        latencies_dict: Dict[str, List[float]],
        title: str = 'Latency Distribution',
        filename: str = 'latency_dist.png'
    ) -> Figure:
        """
        Plot latency distributions.
        
        Args:
            latencies_dict: Dictionary mapping method names to latency lists
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        for method_name, latencies in latencies_dict.items():
            ax1.hist(
                latencies,
                bins=50,
                alpha=0.6,
                label=method_name,
                edgecolor='black'
            )
        
        ax1.set_xlabel('Latency (ms)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Latency Histogram', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data = [latencies_dict[name] for name in latencies_dict.keys()]
        labels = list(latencies_dict.keys())
        
        bp = ax2.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True
        )
        
        # Color boxes
        colors = sns.color_palette('Set2', len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_title('Latency Box Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add 100ms target line
        ax2.axhline(y=100, color='r', linestyle='--', linewidth=1, label='Target (100ms)')
        ax2.legend(fontsize=9)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved latency distribution: {output_path}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, ClassificationMetrics],
        title: str = 'Metrics Comparison',
        filename: str = 'metrics_comparison.png'
    ) -> Figure:
        """
        Plot bar chart comparing metrics across methods.
        
        Args:
            metrics_dict: Dictionary mapping method names to metrics
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        methods = list(metrics_dict.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        
        # Extract values
        values = {
            'Accuracy': [metrics_dict[m].accuracy for m in methods],
            'Precision': [metrics_dict[m].precision for m in methods],
            'Recall': [metrics_dict[m].recall for m in methods],
            'F1': [metrics_dict[m].f1 for m in methods],
            'ROC-AUC': [metrics_dict[m].roc_auc for m in methods],
        }
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.15
        
        for i, metric_name in enumerate(metrics_names):
            offset = width * (i - 2)
            ax.bar(
                x + offset,
                values[metric_name],
                width,
                label=metric_name
            )
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=10, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved metrics comparison: {output_path}")
        
        return fig
    
    def plot_adaptive_drift(
        self,
        timestamps: List[float],
        baseline_values: List[float],
        current_values: List[float],
        title: str = 'Baseline Adaptation Over Time',
        filename: str = 'adaptive_drift.png'
    ) -> Figure:
        """
        Plot baseline adaptation and drift detection.
        
        Args:
            timestamps: Time points
            baseline_values: Baseline values over time
            current_values: Current observed values
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(timestamps, baseline_values, label='Baseline', linewidth=2, marker='o')
        ax.plot(timestamps, current_values, label='Current', linewidth=2, marker='s', alpha=0.7)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved adaptive drift plot: {output_path}")
        
        return fig
    
    def close_all(self):
        """Close all matplotlib figures."""
        plt.close('all')


__all__ = ['Visualizer']