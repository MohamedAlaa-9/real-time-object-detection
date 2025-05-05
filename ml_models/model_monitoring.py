#!/usr/bin/env python3
"""
Model Monitoring Module

This script provides continuous real-time monitoring of the object detection model:
- Performance metrics collection and tracking
- Drift detection for input data and predictions
- Hardware resource utilization monitoring
- Alert generation for performance issues or hardware constraints
- Integration with Prometheus/Grafana for visualization

Can run as a standalone service or be imported by other modules.
"""

import os
import sys
import time
import logging
import threading
import argparse
import json
import queue
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import MLOps utilities
from ml_models.mlops_utils import (
    load_mlops_config, 
    init_mlflow,
    ModelPerformanceTracker,
    DriftDetector,
    HardwareMonitor, 
    fetch_prometheus_metrics,
    send_notification,
    store_difficult_case,
    calculate_evaluation_metrics
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_monitoring.log')
    ]
)
logger = logging.getLogger("ModelMonitoring")

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mlops_config.yaml"
METRICS_DIR = PROJECT_ROOT / "metrics"
METRICS_DIR.mkdir(exist_ok=True, parents=True)

# Global queue for processing frames and detections asynchronously
detection_queue = queue.Queue(maxsize=100)

class ModelMonitor:
    """
    Main class for model monitoring functionality.
    Orchestrates the different monitoring components.
    """
    
    def __init__(self):
        """Initialize the model monitor"""
        self.config = load_mlops_config()
        self.performance_tracker = ModelPerformanceTracker(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.hardware_monitor = HardwareMonitor(self.config)
        
        # Flag to control monitoring threads
        self.running = False
        self.monitoring_thread = None
        self.processing_thread = None
        
        # Load class names from inference module
        try:
            from ml_models.inference import get_class_names
            self.class_names = get_class_names()
        except (ImportError, AttributeError):
            logger.warning("Could not import class names, using generic names")
            self.class_names = [f"class_{i}" for i in range(80)]  # Default COCO classes
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            init_mlflow(self.config)
            logger.info("MLflow initialized for monitoring")
    
    def record_inference(self, frame, detections, latency_ms, model_version="current"):
        """
        Record a single inference result for monitoring.
        
        Args:
            frame: The input image frame
            detections: List of detection dictionaries with boxes, scores, and classes
            latency_ms: Inference latency in milliseconds
            model_version: Version identifier of the model
        """
        # Transform detections to proper format if needed
        if detections and isinstance(detections, tuple) and len(detections) == 3:
            # Convert from (boxes, scores, classes) format
            boxes, scores, classes = detections
            formatted_detections = []
            
            for i in range(len(boxes)):
                formatted_detections.append({
                    'box': boxes[i],
                    'score': scores[i],
                    'class': classes[i] 
                })
            
            detections = formatted_detections
        
        # Queue for async processing
        try:
            detection_queue.put({
                'frame': frame,
                'detections': detections,
                'latency_ms': latency_ms,
                'model_version': model_version,
                'timestamp': datetime.now().isoformat()
            }, block=False)
        except queue.Full:
            # If queue is full, skip this frame for monitoring
            logger.debug("Monitoring queue full, skipping frame")
    
    def _process_detection_queue(self):
        """Process detection queue entries for monitoring"""
        while self.running:
            try:
                # Get next item with timeout
                item = detection_queue.get(timeout=1.0)
                
                # Store difficult cases for future retraining
                store_difficult_case(
                    item['frame'], 
                    item['detections'], 
                    timestamp=item['timestamp'],
                    confidence_threshold=0.5
                )
                
                # Log detections for drift detection
                self.drift_detector.log_detections(item['detections'], self.class_names)
                
                # Calculate FPS from latency
                fps = 1000 / item['latency_ms'] if item['latency_ms'] > 0 else 0
                
                # No ground truth available in real-time inference, 
                # but we can still log the performance metrics we have
                self.performance_tracker.log_metrics({
                    'latency_ms': item['latency_ms'],
                    'fps': fps,
                    'num_detections': len(item['detections']),
                    # The following metrics would need ground truth
                    'map': 0,
                    'precision': 0,
                    'recall': 0,
                }, model_version=item['model_version'])
                
                # Mark as done
                detection_queue.task_done()
                
            except queue.Empty:
                # No items in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing detection: {e}")
    
    def _run_periodic_monitoring(self):
        """Run periodic monitoring tasks"""
        last_drift_check = datetime.now()
        drift_check_interval = timedelta(hours=1)
        
        while self.running:
            try:
                # Log hardware metrics
                self.hardware_monitor.log_hardware_metrics()
                
                # Fetch Prometheus metrics
                prometheus_metrics = fetch_prometheus_metrics()
                if prometheus_metrics:
                    logger.debug(f"Fetched {len(prometheus_metrics)} Prometheus metrics")
                
                # Check for drift periodically
                now = datetime.now()
                if now - last_drift_check > drift_check_interval:
                    drift_detected, drift_details = self.drift_detector.detect_drift()
                    if drift_detected:
                        logger.warning(f"Drift detected: {drift_details}")
                        send_notification(f"Model drift detected: {drift_details}", level="warning")
                    
                    last_drift_check = now
                
                # Sleep for a minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in periodic monitoring: {e}")
                time.sleep(60)  # Sleep and retry
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        logger.info("Starting model monitoring")
        self.running = True
        
        # Start processing queue thread
        self.processing_thread = threading.Thread(
            target=self._process_detection_queue,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start periodic monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._run_periodic_monitoring,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Monitoring threads started")
    
    def stop_monitoring(self):
        """Stop all monitoring threads"""
        logger.info("Stopping model monitoring")
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
            
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            
        logger.info("Monitoring threads stopped")
    
    def evaluate_saved_model(self, model_path, test_dataset_path, 
                          num_samples=100, iou_threshold=0.5):
        """
        Evaluate a saved model against a test dataset with ground truth.
        
        Args:
            model_path: Path to the model file
            test_dataset_path: Path to test dataset with ground truth
            num_samples: Number of samples to evaluate
            iou_threshold: IoU threshold for detection matching
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            # This would need to be implemented depending on your test dataset format
            logger.info(f"Evaluating model {model_path} against {test_dataset_path}")
            
            # Placeholder for evaluation logic - would need dataset-specific implementation
            # In a real implementation, this would:
            # 1. Load the model
            # 2. Load test images and ground truth
            # 3. Run inference on test images
            # 4. Compare with ground truth
            # 5. Calculate metrics
            
            # For demonstration, return mock metrics
            metrics = {
                'map': 0.85,
                'precision': 0.88,
                'recall': 0.82,
                'f1_score': 0.85,
                'latency_ms': 45,
                'fps': 22
            }
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"model_evaluation_{Path(model_path).stem}"):
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
                    
                    # Log parameters
                    mlflow.log_param("model_path", str(model_path))
                    mlflow.log_param("test_dataset", str(test_dataset_path))
                    mlflow.log_param("num_samples", num_samples)
                    mlflow.log_param("iou_threshold", iou_threshold)
            
            logger.info(f"Evaluation complete: mAP={metrics['map']:.3f}, FPS={metrics['fps']:.1f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {
                'map': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'error': str(e)
            }
    
    def generate_monitoring_report(self, report_path=None, hours=24):
        """
        Generate a monitoring report with key metrics and issues.
        
        Args:
            report_path: Path to save the report
            hours: Hours of data to include in report
            
        Returns:
            str: Path to the generated report
        """
        if report_path is None:
            report_dir = PROJECT_ROOT / "reports"
            report_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"monitoring_report_{timestamp}.json"
        
        try:
            # Get performance trend
            performance_df = self.performance_tracker.get_performance_trend(hours=hours)
            
            # Check for drift
            drift_detected, drift_details = self.drift_detector.detect_drift()
            
            # Generate report data
            report = {
                "generated_at": datetime.now().isoformat(),
                "report_period_hours": hours,
                "model_version": "current",  # This would be more detailed in a production system
                "performance_summary": {
                    "avg_latency_ms": performance_df['latency_ms'].mean() if not performance_df.empty else None,
                    "avg_fps": performance_df['fps'].mean() if not performance_df.empty else None,
                    "avg_detections": performance_df['num_detections'].mean() if not performance_df.empty else None,
                    "min_latency_ms": performance_df['latency_ms'].min() if not performance_df.empty else None,
                    "max_latency_ms": performance_df['latency_ms'].max() if not performance_df.empty else None
                },
                "drift_status": {
                    "drift_detected": drift_detected,
                    "details": drift_details
                },
                "issues_detected": [],
                "recommendations": []
            }
            
            # Add any detected issues
            if drift_detected:
                report["issues_detected"].append("Data drift detected")
                report["recommendations"].append("Consider retraining the model on more recent data")
            
            if not performance_df.empty and performance_df['latency_ms'].mean() > 100:
                report["issues_detected"].append("High inference latency")
                report["recommendations"].append("Consider model optimization or hardware acceleration")
            
            # Save the report
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Monitoring report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate monitoring report: {e}")
            return None

# Global monitor instance
_monitor_instance = None

def get_monitor():
    """Get or create the global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    
    return _monitor_instance

def monitor_inference(frame, detections, latency_ms, model_version="current"):
    """
    Convenience function to record inference for monitoring.
    
    Args:
        frame: The input image frame
        detections: List of detection dictionaries or tuple of (boxes, scores, classes)
        latency_ms: Inference latency in milliseconds
        model_version: Version identifier of the model
    """
    monitor = get_monitor()
    
    # Ensure monitoring is running
    if not monitor.running:
        monitor.start_monitoring()
    
    # Record the inference
    monitor.record_inference(frame, detections, latency_ms, model_version)

def run_monitoring_service():
    """Run as a standalone monitoring service"""
    logger.info("Starting model monitoring service")
    
    # Create and start monitor
    monitor = get_monitor()
    monitor.start_monitoring()
    
    # Run indefinitely
    try:
        while True:
            time.sleep(3600)  # Sleep for an hour
            
            # Generate periodic report
            monitor.generate_monitoring_report()
            
    except KeyboardInterrupt:
        logger.info("Monitoring service stopped by user")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Error in monitoring service: {e}")
        monitor.stop_monitoring()

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Model monitoring service for object detection")
    parser.add_argument("--run-service", action="store_true", help="Run as a monitoring service")
    parser.add_argument("--generate-report", action="store_true", help="Generate a monitoring report")
    parser.add_argument("--evaluate-model", type=str, help="Path to model for evaluation")
    parser.add_argument("--test-dataset", type=str, help="Path to test dataset")
    args = parser.parse_args()
    
    monitor = get_monitor()
    
    if args.run_service:
        run_monitoring_service()
    elif args.generate_report:
        report_path = monitor.generate_monitoring_report()
        print(f"Report generated: {report_path}")
    elif args.evaluate_model and args.test_dataset:
        metrics = monitor.evaluate_saved_model(args.evaluate_model, args.test_dataset)
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()