#!/usr/bin/env python3
"""
MLOps Utilities Module

This module provides utility functions for MLOps processes, including:
- Model performance tracking
- MLflow integration
- Drift detection and monitoring
- Model versioning and metadata management
- Evaluation metrics calculation
"""

import os
import sys
import yaml
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Union, Any

# Import MLflow conditionally to handle environments without it
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import optional monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MLOps-Utils")

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mlops_config.yaml"
METRICS_DIR = PROJECT_ROOT / "metrics"
DIFFICULT_CASES_DIR = PROJECT_ROOT / "data" / "difficult_cases"

# Ensure necessary directories exist
METRICS_DIR.mkdir(exist_ok=True, parents=True)
DIFFICULT_CASES_DIR.mkdir(exist_ok=True, parents=True)

# Load MLOps configuration
def load_mlops_config() -> Dict:
    """
    Load MLOps configuration from YAML file.
    
    Returns:
        Dict: Configuration parameters for MLOps pipeline
    """
    try:
        if not CONFIG_PATH.exists():
            logger.error(f"MLOps configuration file not found at {CONFIG_PATH}")
            return {}
            
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded MLOps configuration from {CONFIG_PATH}")
            return config
    except Exception as e:
        logger.error(f"Failed to load MLOps configuration: {e}")
        return {}

# Initialize MLflow
def init_mlflow(config: Dict = None) -> bool:
    """
    Initialize MLflow for experiment tracking.
    
    Args:
        config: MLOps configuration
        
    Returns:
        bool: True if MLflow was successfully initialized
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow is not available. Install with 'pip install mlflow'")
        return False
        
    try:
        if config is None:
            config = load_mlops_config()
            
        mlflow_config = config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri')
        experiment_name = mlflow_config.get('experiment_name', 'object_detection_monitoring')
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to {tracking_uri}")
            
        # Set experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to {experiment_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        return False

# Model Performance Tracking
class ModelPerformanceTracker:
    """Tracks model performance metrics over time"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the performance tracker.
        
        Args:
            config: MLOps configuration
        """
        self.config = config if config else load_mlops_config()
        self.metrics_file = METRICS_DIR / "performance_metrics.csv"
        self.threshold_metrics = self.config.get('monitoring', {}).get('performance_threshold', {})
        
        # Create metrics file with headers if it doesn't exist
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("timestamp,map,precision,recall,latency_ms,fps,num_detections,model_version\n")
    
    def log_metrics(self, metrics: Dict[str, float], model_version: str = "current") -> None:
        """
        Log performance metrics to CSV and optionally MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            model_version: Version identifier of the model
        """
        # Ensure all required metrics exist
        required_metrics = {'map', 'precision', 'recall', 'latency_ms', 'fps', 'num_detections'}
        for metric in required_metrics:
            if metric not in metrics:
                metrics[metric] = 0.0
                
        # Add timestamp
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{metrics['map']},{metrics['precision']},"
                    f"{metrics['recall']},{metrics['latency_ms']},"
                    f"{metrics['fps']},{metrics['num_detections']},{model_version}\n")
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"monitoring_{model_version}", nested=True):
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
        
        # Check for performance issues
        self._check_performance_thresholds(metrics)
                
    def _check_performance_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Check if metrics fall below defined thresholds.
        
        Args:
            metrics: Dictionary of current metrics
        
        Returns:
            bool: True if any metrics are below thresholds
        """
        issues_found = False
        
        # Check map
        min_map = self.threshold_metrics.get('min_map', 0.0)
        if metrics.get('map', 1.0) < min_map:
            logger.warning(f"MAP below threshold: {metrics.get('map', 0):.3f} < {min_map}")
            issues_found = True
        
        # Check precision
        min_precision = self.threshold_metrics.get('min_precision', 0.0)
        if metrics.get('precision', 1.0) < min_precision:
            logger.warning(f"Precision below threshold: {metrics.get('precision', 0):.3f} < {min_precision}")
            issues_found = True
            
        # Check recall
        min_recall = self.threshold_metrics.get('min_recall', 0.0)
        if metrics.get('recall', 1.0) < min_recall:
            logger.warning(f"Recall below threshold: {metrics.get('recall', 0):.3f} < {min_recall}")
            issues_found = True
        
        # Check latency
        max_latency = self.threshold_metrics.get('max_latency_ms', float('inf'))
        if metrics.get('latency_ms', 0) > max_latency:
            logger.warning(f"Latency above threshold: {metrics.get('latency_ms', 0):.2f}ms > {max_latency}ms")
            issues_found = True
            
        return issues_found
    
    def get_performance_trend(self, hours: int = 24) -> pd.DataFrame:
        """
        Get performance trend over the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            pd.DataFrame: DataFrame with performance metrics
        """
        try:
            # Read metrics data
            if not self.metrics_file.exists():
                return pd.DataFrame()
                
            df = pd.read_csv(self.metrics_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for the specified time period
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_df = df[df['timestamp'] >= cutoff_time]
            
            return recent_df
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            return pd.DataFrame()
    
    def detect_performance_drop(self) -> Tuple[bool, Dict]:
        """
        Detect if there's been a significant drop in performance.
        
        Returns:
            Tuple[bool, Dict]: (drop_detected, drop_details)
        """
        config = self.config.get('retraining', {}).get('triggers', {}).get('performance_drop', {})
        threshold = config.get('threshold', 0.1)
        window_hours = config.get('window_hours', 24)
        
        # Get recent performance data
        df = self.get_performance_trend(hours=window_hours)
        if df.empty or len(df) < 2:
            return False, {}
            
        # Calculate rolling averages to smooth out noise
        window_size = min(10, max(2, len(df) // 10))
        df['map_rolling'] = df['map'].rolling(window=window_size, min_periods=1).mean()
        df['latency_rolling'] = df['latency_ms'].rolling(window=window_size, min_periods=1).mean()
        
        # Get first and last values for key metrics
        first_map = df['map_rolling'].iloc[0]
        last_map = df['map_rolling'].iloc[-1]
        map_change = (last_map - first_map) / first_map if first_map > 0 else 0
        
        # Check if drop exceeds threshold
        drop_detected = map_change < -threshold
        
        if drop_detected:
            logger.warning(f"Performance drop detected: MAP changed by {map_change:.2%}")
            
        return drop_detected, {
            'map_change': map_change,
            'first_map': first_map,
            'last_map': last_map,
            'window_hours': window_hours
        }

# Drift Detection
class DriftDetector:
    """Detects drift in model inputs and predictions over time"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize drift detector.
        
        Args:
            config: MLOps configuration
        """
        self.config = config if config else load_mlops_config()
        self.drift_config = self.config.get('monitoring', {}).get('drift_detection', {})
        self.enabled = self.drift_config.get('enabled', False)
        self.threshold = self.drift_config.get('threshold', 0.15)
        self.detection_window = self.drift_config.get('detection_window', 24)
        
        self.class_dist_file = METRICS_DIR / "class_distribution.csv"
        self.detections_file = METRICS_DIR / "detection_stats.csv"
        self.consecutive_alerts = 0
        
        # Initialize files if they don't exist
        if not self.class_dist_file.exists():
            with open(self.class_dist_file, 'w') as f:
                f.write("timestamp,class_id,class_name,count,proportion\n")
                
        if not self.detections_file.exists():
            with open(self.detections_file, 'w') as f:
                f.write("timestamp,num_detections,avg_confidence,avg_box_size\n")
    
    def log_detections(self, detections: List[Dict], class_names: List[str]) -> None:
        """
        Log detection statistics for drift monitoring.
        
        Args:
            detections: List of detection dictionaries with 'box', 'score', 'class'
            class_names: List of class names for the model
        """
        if not self.enabled or not detections:
            return
            
        timestamp = datetime.now().isoformat()
        
        # Calculate detection statistics
        num_detections = len(detections)
        avg_confidence = np.mean([det.get('score', 0) for det in detections])
        
        # Calculate average box size (normalized)
        avg_box_size = 0
        if num_detections > 0:
            box_sizes = []
            for det in detections:
                box = det.get('box', [0, 0, 0, 0])
                # Calculate box area (width * height)
                width = box[2] - box[0]
                height = box[3] - box[1] 
                area = width * height
                box_sizes.append(area)
            avg_box_size = np.mean(box_sizes)
        
        # Log detection statistics
        with open(self.detections_file, 'a') as f:
            f.write(f"{timestamp},{num_detections},{avg_confidence},{avg_box_size}\n")
            
        # Log class distribution
        class_counts = {}
        for det in detections:
            class_id = det.get('class', 0)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
        # Calculate proportions
        total = sum(class_counts.values())
        with open(self.class_dist_file, 'a') as f:
            for class_id, count in class_counts.items():
                proportion = count / total if total > 0 else 0
                class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
                f.write(f"{timestamp},{class_id},{class_name},{count},{proportion}\n")
    
    def detect_drift(self) -> Tuple[bool, Dict]:
        """
        Detect if there's been a significant drift in detection patterns.
        
        Returns:
            Tuple[bool, Dict]: (drift_detected, drift_details)
        """
        if not self.enabled:
            return False, {}
            
        try:
            # Check if we have enough data
            if not self.class_dist_file.exists() or not self.detections_file.exists():
                return False, {}
                
            # Load detection stats data
            det_df = pd.read_csv(self.detections_file)
            det_df['timestamp'] = pd.to_datetime(det_df['timestamp'])
            
            # Load class distribution data
            class_df = pd.read_csv(self.class_dist_file)
            class_df['timestamp'] = pd.to_datetime(class_df['timestamp'])
            
            # Filter for the specified time window
            cutoff_time = datetime.now() - timedelta(hours=self.detection_window)
            recent_det_df = det_df[det_df['timestamp'] >= cutoff_time]
            recent_class_df = class_df[class_df['timestamp'] >= cutoff_time]
            
            # If insufficient data, return no drift
            if len(recent_det_df) < 10 or len(recent_class_df) < 10:
                return False, {'reason': 'insufficient_data'}
                
            # Check for drift in detection count and confidence
            det_count_drift = self._check_metric_drift(recent_det_df, 'num_detections')
            confidence_drift = self._check_metric_drift(recent_det_df, 'avg_confidence')
            
            # Check for drift in class distribution
            class_dist_drift = self._check_class_distribution_drift(recent_class_df)
            
            # Combine drift signals
            drift_detected = det_count_drift or confidence_drift or class_dist_drift
            
            if drift_detected:
                self.consecutive_alerts += 1
                logger.warning(f"Drift detected! Consecutive alerts: {self.consecutive_alerts}")
            else:
                self.consecutive_alerts = 0
                
            return drift_detected, {
                'detection_count_drift': det_count_drift,
                'confidence_drift': confidence_drift,
                'class_distribution_drift': class_dist_drift,
                'consecutive_alerts': self.consecutive_alerts
            }
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False, {'error': str(e)}
    
    def _check_metric_drift(self, df: pd.DataFrame, column: str) -> bool:
        """
        Check if a metric has drifted over time.
        
        Args:
            df: DataFrame with metric data
            column: Column to check for drift
            
        Returns:
            bool: True if drift detected
        """
        if df.empty or len(df) < 10:
            return False
            
        # Split into reference and current windows
        half_idx = len(df) // 2
        reference_window = df.iloc[:half_idx][column].values
        current_window = df.iloc[half_idx:][column].values
        
        # Basic drift detection using distribution statistics
        ref_mean = np.mean(reference_window)
        cur_mean = np.mean(current_window)
        
        # Calculate relative change
        rel_change = abs((cur_mean - ref_mean) / (ref_mean if ref_mean != 0 else 1))
        
        return rel_change > self.threshold
    
    def _check_class_distribution_drift(self, df: pd.DataFrame) -> bool:
        """
        Check if class distribution has drifted.
        
        Args:
            df: DataFrame with class distribution data
            
        Returns:
            bool: True if distribution drift detected
        """
        if df.empty or len(df) < 10:
            return False
            
        # Get unique timestamps, sorted
        timestamps = sorted(df['timestamp'].unique())
        if len(timestamps) < 4:  # Need enough time points for comparison
            return False
            
        # Get reference and current distribution from start and end periods
        ref_period = df[df['timestamp'] <= timestamps[len(timestamps)//3]]
        cur_period = df[df['timestamp'] >= timestamps[2*len(timestamps)//3]]
        
        # Aggregate class proportions
        ref_dist = ref_period.groupby('class_id')['proportion'].mean().reset_index()
        cur_dist = cur_period.groupby('class_id')['proportion'].mean().reset_index()
        
        # Create a complete distribution for both periods
        all_classes = set(ref_dist['class_id'].tolist() + cur_dist['class_id'].tolist())
        
        ref_props = {cls_id: 0 for cls_id in all_classes}
        cur_props = {cls_id: 0 for cls_id in all_classes}
        
        # Fill reference distribution
        for _, row in ref_dist.iterrows():
            ref_props[row['class_id']] = row['proportion']
            
        # Fill current distribution
        for _, row in cur_dist.iterrows():
            cur_props[row['class_id']] = row['proportion']
            
        # Calculate Jensen-Shannon divergence (symmetric measure of difference between distributions)
        # First, convert to numpy arrays
        ref_array = np.array([ref_props[cls_id] for cls_id in sorted(all_classes)])
        cur_array = np.array([cur_props[cls_id] for cls_id in sorted(all_classes)])
        
        # Normalize if needed
        ref_array = ref_array / ref_array.sum() if ref_array.sum() > 0 else ref_array
        cur_array = cur_array / cur_array.sum() if cur_array.sum() > 0 else cur_array
        
        # Calculate JS divergence
        m_array = 0.5 * (ref_array + cur_array)
        js_div = 0.5 * self._kl_divergence(ref_array, m_array) + 0.5 * self._kl_divergence(cur_array, m_array)
        
        return js_div > self.threshold
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions.
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            float: KL divergence
        """
        # Avoid division by zero and log(0)
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        
        return np.sum(p * np.log(p / q))

# Hardware Monitoring
class HardwareMonitor:
    """Monitors hardware utilization and performance"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize hardware monitor.
        
        Args:
            config: MLOps configuration
        """
        self.config = config if config else load_mlops_config()
        self.hw_config = self.config.get('hardware_monitoring', {})
        self.enabled = self.hw_config.get('enabled', False)
        self.metrics_file = METRICS_DIR / "hardware_metrics.csv"
        
        # Initialize metrics file if it doesn't exist
        if self.enabled and not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("timestamp,cpu_percent,memory_percent,gpu_memory_percent,gpu_utilization\n")
    
    def log_hardware_metrics(self) -> Dict[str, float]:
        """
        Log current hardware utilization metrics.
        
        Returns:
            Dict[str, float]: Dictionary of hardware metrics
        """
        if not self.enabled:
            return {}
            
        metrics = {}
        timestamp = datetime.now().isoformat()
        
        # CPU and memory metrics (if psutil is available)
        if PSUTIL_AVAILABLE:
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['memory_percent'] = psutil.virtual_memory().percent
        else:
            metrics['cpu_percent'] = -1
            metrics['memory_percent'] = -1
            
        # GPU metrics (if pynvml is available)
        if PYNVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics['gpu_memory_percent'] = (info.used / info.total) * 100
                    metrics['gpu_utilization'] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                else:
                    metrics['gpu_memory_percent'] = -1
                    metrics['gpu_utilization'] = -1
            except Exception:
                metrics['gpu_memory_percent'] = -1
                metrics['gpu_utilization'] = -1
        else:
            metrics['gpu_memory_percent'] = -1
            metrics['gpu_utilization'] = -1
            
        # Log to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{metrics.get('cpu_percent', -1)},"
                    f"{metrics.get('memory_percent', -1)},"
                    f"{metrics.get('gpu_memory_percent', -1)},"
                    f"{metrics.get('gpu_utilization', -1)}\n")
        
        # Check for hardware issues
        self._check_hardware_thresholds(metrics)
        
        return metrics
    
    def _check_hardware_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Check if hardware metrics exceed defined thresholds.
        
        Args:
            metrics: Dictionary of hardware metrics
            
        Returns:
            bool: True if any metrics exceed thresholds
        """
        issues_found = False
        
        # Check CPU utilization
        cpu_threshold = self.hw_config.get('alerts', {}).get('cpu_utilization_threshold', 95)
        if metrics.get('cpu_percent', 0) > cpu_threshold:
            logger.warning(f"High CPU utilization: {metrics.get('cpu_percent', 0):.1f}% > {cpu_threshold}%")
            issues_found = True
            
        # Check GPU memory
        gpu_mem_threshold = self.hw_config.get('alerts', {}).get('gpu_memory_threshold', 90)
        if metrics.get('gpu_memory_percent', 0) > gpu_mem_threshold:
            logger.warning(f"High GPU memory usage: {metrics.get('gpu_memory_percent', 0):.1f}% > {gpu_mem_threshold}%")
            issues_found = True
            
        return issues_found

# Store Difficult Cases
def store_difficult_case(frame, detections: List[Dict], timestamp: str = None,
                        confidence_threshold: float = 0.5) -> None:
    """
    Store frames with low confidence detections for later analysis and retraining.
    
    Args:
        frame: The image frame as numpy array
        detections: List of detections with boxes, scores, and classes
        timestamp: Optional timestamp string
        confidence_threshold: Threshold below which to consider a detection difficult
    """
    config = load_mlops_config()
    if not config.get('data_collection', {}).get('store_difficult_cases', False):
        return
        
    # Check if any detection has low confidence
    has_difficult_case = any(det.get('score', 1.0) < confidence_threshold for det in detections)
    
    if not has_difficult_case:
        return
        
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        
    # Save image with detections
    try:
        import cv2
        
        # Create directory if it doesn't exist
        save_dir = DIFFICULT_CASES_DIR / datetime.now().strftime("%Y-%m-%d")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save original frame
        image_path = save_dir / f"difficult_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        
        # Save detection metadata
        meta_path = save_dir / f"difficult_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'detections': [
                    {
                        'box': det.get('box', []),
                        'score': det.get('score', 0),
                        'class': det.get('class', 0)
                    } for det in detections
                ]
            }, f)
            
        logger.debug(f"Stored difficult case: {image_path}")
    except Exception as e:
        logger.error(f"Failed to store difficult case: {e}")

# Calculate Evaluation Metrics
def calculate_evaluation_metrics(predictions: List[Dict], ground_truth: List[Dict],
                               iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate evaluation metrics like precision, recall, F1, and mAP.
    
    Args:
        predictions: List of prediction dictionaries with 'box', 'score', 'class'
        ground_truth: List of ground truth dictionaries with 'box', 'class'
        iou_threshold: IoU threshold for matching predictions to ground truth
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Initialize metrics
    metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'map': 0.0
    }
    
    # Count TP, FP, FN
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Keep track of matched ground truth boxes
    matched_gt = set()
    
    # Sort predictions by confidence score (high to low)
    preds_sorted = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
    
    for pred in preds_sorted:
        pred_box = pred.get('box', [0, 0, 0, 0])
        pred_class = pred.get('class', 0)
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find the ground truth box with the highest IoU
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue  # Skip already matched ground truth boxes
                
            gt_box = gt.get('box', [0, 0, 0, 0])
            gt_class = gt.get('class', 0)
            
            # Only consider same class
            if pred_class != gt_class:
                continue
                
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # Check if we have a match
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
    
    # Count false negatives (unmatched ground truth boxes)
    false_negatives = len(ground_truth) - len(matched_gt)
    
    # Calculate precision and recall
    if true_positives + false_positives > 0:
        metrics['precision'] = true_positives / (true_positives + false_positives)
    else:
        metrics['precision'] = 0
        
    if true_positives + false_negatives > 0:
        metrics['recall'] = true_positives / (true_positives + false_negatives)
    else:
        metrics['recall'] = 0
        
    # Calculate F1 score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
        
    # Calculate mAP (simplified version for this application)
    metrics['map'] = metrics['precision']  # Replace with proper mAP calculation
    
    return metrics

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Unpack boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

# Check if retraining is needed
def check_retraining_needed() -> Tuple[bool, str]:
    """
    Check if model retraining is needed based on performance and drift.
    
    Returns:
        Tuple[bool, str]: (retraining_needed, reason)
    """
    config = load_mlops_config()
    retraining_config = config.get('retraining', {})
    
    # Check for performance drop
    performance_tracker = ModelPerformanceTracker(config)
    performance_drop, _ = performance_tracker.detect_performance_drop()
    
    if (performance_drop and 
        retraining_config.get('triggers', {}).get('performance_drop', {}).get('enabled', False)):
        return True, "performance_drop"
    
    # Check for drift
    drift_detector = DriftDetector(config)
    drift_detected, drift_details = drift_detector.detect_drift()
    consecutive_alerts = drift_details.get('consecutive_alerts', 0)
    
    drift_threshold = retraining_config.get('triggers', {}).get('drift_detected', {}).get('consecutive_alerts', 3)
    
    if (drift_detected and consecutive_alerts >= drift_threshold and 
        retraining_config.get('triggers', {}).get('drift_detected', {}).get('enabled', False)):
        return True, "drift_detected"
    
    return False, "none"

# Fetch metrics from Prometheus
def fetch_prometheus_metrics(metrics_endpoint: str = None) -> Dict[str, float]:
    """
    Fetch metrics from Prometheus endpoint.
    
    Args:
        metrics_endpoint: URL of the Prometheus metrics endpoint
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    config = load_mlops_config()
    if metrics_endpoint is None:
        metrics_endpoint = config.get('monitoring', {}).get('metrics_endpoint', 'http://localhost:8001')
        
    try:
        response = requests.get(metrics_endpoint, timeout=5)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch metrics: {response.status_code}")
            return {}
            
        # Parse metrics text
        metrics = {}
        for line in response.text.splitlines():
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Parse metric name and value
            parts = line.split(' ')
            if len(parts) < 2:
                continue
                
            name = parts[0]
            try:
                value = float(parts[1])
                metrics[name] = value
            except ValueError:
                continue
                
        return metrics
    except Exception as e:
        logger.error(f"Error fetching Prometheus metrics: {e}")
        return {}

# Notify about issues (email, Slack, etc.)
def send_notification(message: str, level: str = "warning") -> bool:
    """
    Send a notification about model issues.
    
    Args:
        message: Notification message
        level: Severity level (info, warning, error)
        
    Returns:
        bool: True if notification was sent successfully
    """
    config = load_mlops_config()
    notifications_config = config.get('notifications', {})
    
    # Log the notification
    if level == "info":
        logger.info(f"NOTIFICATION: {message}")
    elif level == "warning":
        logger.warning(f"NOTIFICATION: {message}")
    else:
        logger.error(f"NOTIFICATION: {message}")
    
    # Email notification
    if notifications_config.get('email', {}).get('enabled', False):
        try:
            # Simplified email sending (replace with proper implementation)
            recipients = notifications_config.get('email', {}).get('recipients', [])
            logger.info(f"Would send email to {recipients}: {message}")
            # Implement actual email sending here
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    # Slack notification
    if notifications_config.get('slack', {}).get('enabled', False):
        try:
            webhook_url = notifications_config.get('slack', {}).get('webhook_url', '')
            if webhook_url:
                payload = {"text": f"[{level.upper()}] {message}"}
                requests.post(webhook_url, json=payload)
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    return True

# If script is run directly, test MLOps utilities
if __name__ == "__main__":
    print("Testing MLOps Utilities")
    
    # Load config
    config = load_mlops_config()
    print(f"Loaded configuration: {bool(config)}")
    
    # Initialize MLflow if available
    mlflow_initialized = init_mlflow(config)
    print(f"MLflow initialized: {mlflow_initialized}")
    
    # Test hardware monitoring
    hw_monitor = HardwareMonitor(config)
    hw_metrics = hw_monitor.log_hardware_metrics()
    print(f"Hardware metrics: {hw_metrics}")
    
    print("All tests completed.")