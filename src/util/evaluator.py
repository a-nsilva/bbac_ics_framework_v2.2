#!/usr/bin/env python3
"""
BBAC ICS Framework - Evaluation Metrics

Comprehensive metrics calculation for research paper:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Performance metrics (latency, throughput)
- Statistical tests (t-test, effect size, confidence intervals)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

from .data_structures import (
    ClassificationMetrics,
    LatencyMetrics,
    PerformanceMetrics,
    StatisticalTest,
)


logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluates system performance with comprehensive metrics.
    
    For research paper with IF>7 standards.
    """
    
    def __init__(self):
        """Initialize metrics evaluator."""
        logger.info("MetricsEvaluator initialized")
    
    def evaluate_classification(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_scores: List[float] = None
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.
        
        Args:
            y_true: Ground truth labels ('grant' or 'deny')
            y_pred: Predicted labels ('grant' or 'deny')
            y_scores: Optional confidence scores for ROC/PR curves
            
        Returns:
            ClassificationMetrics dataclass
        """
        # Convert to binary (grant=1, deny=0)
        y_true_binary = [1 if y == 'grant' else 0 for y in y_true]
        y_pred_binary = [1 if y == 'grant' else 0 for y in y_pred]
        
        # Basic metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # ROC-AUC and curves (if scores provided)
        roc_auc = 0.0
        avg_precision = 0.0
        fpr_list = []
        tpr_list = []
        precision_curve = []
        recall_curve = []
        
        if y_scores is not None and len(y_scores) == len(y_true_binary):
            try:
                roc_auc = roc_auc_score(y_true_binary, y_scores)
                avg_precision = average_precision_score(y_true_binary, y_scores)
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
                
                # Precision-Recall curve
                precision_vals, recall_vals, _ = precision_recall_curve(
                    y_true_binary, y_scores
                )
                precision_curve = precision_vals.tolist()
                recall_curve = recall_vals.tolist()
                
            except Exception as e:
                logger.warning(f"Error calculating ROC/PR curves: {e}")
        
        metrics = ClassificationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc),
            avg_precision=float(avg_precision),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            fpr=fpr_list,
            tpr=tpr_list,
            precision_curve=precision_curve,
            recall_curve=recall_curve,
        )
        
        logger.info(
            f"Classification metrics: accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
        )
        
        return metrics
    
    def evaluate_latency(
        self,
        latencies: List[float]
    ) -> LatencyMetrics:
        """
        Calculate latency statistics.
        
        Args:
            latencies: List of latency values in milliseconds
            
        Returns:
            LatencyMetrics dataclass
        """
        latencies_array = np.array(latencies)
        
        metrics = LatencyMetrics(
            mean=float(np.mean(latencies_array)),
            std=float(np.std(latencies_array)),
            p50=float(np.percentile(latencies_array, 50)),
            p95=float(np.percentile(latencies_array, 95)),
            p99=float(np.percentile(latencies_array, 99)),
            values=latencies,
        )
        
        logger.info(
            f"Latency metrics: mean={metrics.mean:.2f}ms, "
            f"p95={metrics.p95:.2f}ms, p99={metrics.p99:.2f}ms"
        )
        
        return metrics
    
    def evaluate_performance(
        self,
        latencies: List[float],
        total_time: float
    ) -> PerformanceMetrics:
        """
        Calculate overall performance metrics.
        
        Args:
            latencies: List of latency values in milliseconds
            total_time: Total execution time in seconds
            
        Returns:
            PerformanceMetrics dataclass
        """
        latency_metrics = self.evaluate_latency(latencies)
        
        total_requests = len(latencies)
        throughput = total_requests / total_time if total_time > 0 else 0.0
        
        metrics = PerformanceMetrics(
            latency=latency_metrics,
            throughput=float(throughput),
            total_requests=total_requests,
            total_time=float(total_time),
        )
        
        logger.info(
            f"Performance metrics: throughput={throughput:.2f} req/s, "
            f"total_requests={total_requests}, total_time={total_time:.2f}s"
        )
        
        return metrics
    
    def compare_methods(
        self,
        method_a_metrics: ClassificationMetrics,
        method_b_metrics: ClassificationMetrics,
        method_a_latencies: List[float],
        method_b_latencies: List[float],
        alpha: float = 0.05
    ) -> Dict[str, StatisticalTest]:
        """
        Statistical comparison between two methods.
        
        Args:
            method_a_metrics: Classification metrics for method A
            method_b_metrics: Classification metrics for method B
            method_a_latencies: Latencies for method A
            method_b_latencies: Latencies for method B
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary of statistical tests for each metric
        """
        comparisons = {}
        
        # Compare F1 scores (using confusion matrix to estimate variance)
        comparisons['f1'] = self._compare_proportions(
            method_a_metrics.f1,
            method_b_metrics.f1,
            method_a_metrics.tp + method_a_metrics.fp + method_a_metrics.fn + method_a_metrics.tn,
            method_b_metrics.tp + method_b_metrics.fp + method_b_metrics.fn + method_b_metrics.tn,
            alpha
        )
        
        # Compare latencies (t-test)
        comparisons['latency'] = self._ttest_comparison(
            method_a_latencies,
            method_b_latencies,
            alpha
        )
        
        logger.info(
            f"Method comparison: F1 p-value={comparisons['f1'].p_value:.4f}, "
            f"Latency p-value={comparisons['latency'].p_value:.4f}"
        )
        
        return comparisons
    
    def _ttest_comparison(
        self,
        group_a: List[float],
        group_b: List[float],
        alpha: float
    ) -> StatisticalTest:
        """Perform t-test comparison between two groups."""
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b)
        
        # Cohen's d effect size
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        std_a = np.std(group_a, ddof=1)
        std_b = np.std(group_b, ddof=1)
        
        pooled_std = np.sqrt(((len(group_a) - 1) * std_a**2 + 
                              (len(group_b) - 1) * std_b**2) / 
                             (len(group_a) + len(group_b) - 2))
        
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # 95% confidence interval for mean difference
        se_diff = pooled_std * np.sqrt(1/len(group_a) + 1/len(group_b))
        df = len(group_a) + len(group_b) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mean_diff = mean_a - mean_b
        ci_low = mean_diff - t_critical * se_diff
        ci_high = mean_diff + t_critical * se_diff
        
        return StatisticalTest(
            p_value=float(p_value),
            effect_size=float(effect_size),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            significant=bool(p_value < alpha),
        )
    
    def _compare_proportions(
        self,
        prop_a: float,
        prop_b: float,
        n_a: int,
        n_b: int,
        alpha: float
    ) -> StatisticalTest:
        """Compare two proportions (e.g., F1 scores)."""
        # Z-test for proportions
        p_pooled = (prop_a * n_a + prop_b * n_b) / (n_a + n_b)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
        
        if se > 0:
            z_stat = (prop_a - prop_b) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            p_value = 1.0
        
        # Effect size (Cohen's h for proportions)
        effect_size = 2 * (np.arcsin(np.sqrt(prop_a)) - np.arcsin(np.sqrt(prop_b)))
        
        # 95% confidence interval
        se_diff = np.sqrt(prop_a * (1 - prop_a) / n_a + 
                          prop_b * (1 - prop_b) / n_b)
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        diff = prop_a - prop_b
        ci_low = diff - z_critical * se_diff
        ci_high = diff + z_critical * se_diff
        
        return StatisticalTest(
            p_value=float(p_value),
            effect_size=float(effect_size),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            significant=bool(p_value < alpha),
        )
    
    def calculate_drift_metrics(
        self,
        baseline_data: List[float],
        current_data: List[float]
    ) -> Dict[str, float]:
        """
        Calculate drift metrics for ADAPTIVE evaluation.
        
        Args:
            baseline_data: Historical baseline values
            current_data: Current values
            
        Returns:
            Dictionary with drift metrics
        """
        # KS test for distribution shift
        ks_stat, ks_pvalue = stats.ks_2samp(baseline_data, current_data)
        
        # Mean shift
        mean_shift = np.mean(current_data) - np.mean(baseline_data)
        
        # Relative change
        baseline_mean = np.mean(baseline_data)
        relative_change = (mean_shift / baseline_mean * 100) if baseline_mean != 0 else 0.0
        
        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'mean_shift': float(mean_shift),
            'relative_change_percent': float(relative_change),
            'drift_detected': bool(ks_pvalue < 0.05),
        }


__all__ = ['MetricsEvaluator']