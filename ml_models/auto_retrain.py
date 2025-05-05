#!/usr/bin/env python3
"""
Auto-Retraining Module

This script implements automated model retraining functionality:
- Scheduled retraining based on time intervals
- Performance-triggered retraining when model performance degrades
- Drift-triggered retraining when significant data or concept drift is detected

The module can be run as a standalone script or as part of a larger monitoring system.
"""

import os
import sys
import time
import yaml
import logging
import argparse
import schedule
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import MLOps utilities
from ml_models.mlops_utils import (
    load_mlops_config, 
    init_mlflow, 
    check_retraining_needed,
    send_notification,
    fetch_prometheus_metrics
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
        logging.FileHandler('auto_retrain.log')
    ]
)
logger = logging.getLogger("AutoRetrain")

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mlops_config.yaml"
TRAIN_SCRIPT = BASE_DIR / "train_yolo.py"
EXPORT_SCRIPT = BASE_DIR / "export_model.py"
VERIFICATION_SCRIPT = BASE_DIR / "verify_model_pipeline.py"
LOCKS_DIR = PROJECT_ROOT / "locks"
RETRAIN_LOCK = LOCKS_DIR / "retrain.lock"

# Ensure locks directory exists
LOCKS_DIR.mkdir(exist_ok=True, parents=True)

def is_retraining_in_progress():
    """Check if a retraining process is already running"""
    return RETRAIN_LOCK.exists()

def set_retraining_lock(metadata=None):
    """Create a lock file to indicate retraining is in progress"""
    if metadata is None:
        metadata = {}
    
    metadata["timestamp"] = datetime.now().isoformat()
    metadata["pid"] = os.getpid()
    
    with open(RETRAIN_LOCK, 'w') as f:
        json.dump(metadata, f)
    
    logger.info(f"Retraining lock created: {RETRAIN_LOCK}")

def remove_retraining_lock():
    """Remove the retraining lock file"""
    if RETRAIN_LOCK.exists():
        RETRAIN_LOCK.unlink()
        logger.info("Retraining lock removed")

def retrain_model(trigger_reason="scheduled"):
    """
    Execute the full model retraining pipeline:
    1. Verify dependencies and dataset
    2. Run training
    3. Export model to optimized format
    4. Verify model loading
    
    Args:
        trigger_reason: Reason for triggering retraining
    
    Returns:
        bool: True if retraining was successful
    """
    # Check if already running
    if is_retraining_in_progress():
        logger.warning("A retraining process is already running. Skipping.")
        return False
    
    # Set lock with metadata
    set_retraining_lock({
        "trigger_reason": trigger_reason,
        "started_at": datetime.now().isoformat()
    })
    
    try:
        start_time = time.time()
        logger.info(f"Starting model retraining (trigger: {trigger_reason})")
        
        # Verify dataset preparation
        logger.info("Checking dataset preparation...")
        subprocess.run(
            ["python", str(PROJECT_ROOT / "datasets/preprocess_datasets.py")],
            check=True
        )
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                config = load_mlops_config()
                init_mlflow(config)
                
                with mlflow.start_run(run_name=f"auto_retrain_{trigger_reason}"):
                    mlflow.log_param("trigger_reason", trigger_reason)
                    mlflow.log_param("retrain_time", datetime.now().isoformat())
                    
                    # Run training script
                    logger.info("Starting model training...")
                    train_result = subprocess.run(
                        ["python", str(TRAIN_SCRIPT)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Log training output
                    mlflow.log_text(train_result.stdout, "train_output.log")
                    if train_result.stderr:
                        mlflow.log_text(train_result.stderr, "train_errors.log")
                    
                    # Export model
                    logger.info("Exporting trained model...")
                    export_result = subprocess.run(
                        ["python", str(EXPORT_SCRIPT)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Log export output
                    mlflow.log_text(export_result.stdout, "export_output.log")
                    if export_result.stderr:
                        mlflow.log_text(export_result.stderr, "export_errors.log")
                    
                    # Verify model pipeline
                    logger.info("Verifying model pipeline...")
                    verify_result = subprocess.run(
                        ["python", str(VERIFICATION_SCRIPT)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Log verification output
                    mlflow.log_text(verify_result.stdout, "verification_output.log")
                    if verify_result.stderr:
                        mlflow.log_text(verify_result.stderr, "verification_errors.log")
                    
                    # Calculate and log total duration
                    duration = time.time() - start_time
                    mlflow.log_metric("retraining_duration_seconds", duration)
                    
                    # Log to MLflow that retraining completed successfully
                    mlflow.log_param("retraining_status", "success")
                    
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")
                # Continue with training even if MLflow fails
                run_training_without_mlflow(trigger_reason)
        else:
            # Run without MLflow
            run_training_without_mlflow(trigger_reason)
        
        logger.info(f"Model retraining completed successfully in {time.time() - start_time:.2f} seconds")
        send_notification(
            f"Model retraining completed successfully (trigger: {trigger_reason})",
            level="info"
        )
        return True
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        send_notification(
            f"Model retraining failed: {str(e)} (trigger: {trigger_reason})",
            level="error"
        )
        return False
    finally:
        # Always remove the lock when done
        remove_retraining_lock()

def run_training_without_mlflow(trigger_reason):
    """Run the training pipeline without MLflow logging"""
    # Run training script
    logger.info("Starting model training (without MLflow)...")
    subprocess.run(
        ["python", str(TRAIN_SCRIPT)],
        check=True
    )
    
    # Export model
    logger.info("Exporting trained model...")
    subprocess.run(
        ["python", str(EXPORT_SCRIPT)],
        check=True
    )
    
    # Verify model pipeline
    logger.info("Verifying model pipeline...")
    subprocess.run(
        ["python", str(VERIFICATION_SCRIPT)],
        check=True
    )

def schedule_retraining_job():
    """Setup scheduled retraining based on configuration"""
    config = load_mlops_config()
    schedule_config = config.get('retraining', {}).get('schedule', {})
    
    if not schedule_config.get('enabled', False):
        logger.info("Scheduled retraining is disabled in configuration")
        return
    
    # Get cron expression from config
    cron_expression = schedule_config.get('cron_expression', '0 0 * * 0')  # Default: weekly at midnight on Sunday
    
    # Parse cron expression
    parts = cron_expression.split()
    if len(parts) != 5:
        logger.error(f"Invalid cron expression: {cron_expression}")
        return
    
    minute, hour, day_of_month, month, day_of_week = parts
    
    # Schedule based on cron expression
    if day_of_week != '*' and day_of_week != '?':
        # Schedule on specific days of week
        days = {
            '0': schedule.every().sunday,
            '1': schedule.every().monday,
            '2': schedule.every().tuesday,
            '3': schedule.every().wednesday,
            '4': schedule.every().thursday,
            '5': schedule.every().friday,
            '6': schedule.every().saturday,
        }
        
        for day_num in day_of_week.split(','):
            if day_num in days:
                days[day_num].at(f"{hour.zfill(2)}:{minute.zfill(2)}").do(retrain_model)
    else:
        # Schedule every day
        schedule.every().day.at(f"{hour.zfill(2)}:{minute.zfill(2)}").do(retrain_model)
    
    logger.info(f"Scheduled retraining job with cron expression: {cron_expression}")

def check_for_retraining_triggers():
    """Check if any retraining triggers are met"""
    needed, reason = check_retraining_needed()
    
    if needed:
        logger.info(f"Retraining trigger detected: {reason}")
        retrain_model(trigger_reason=reason)
    else:
        logger.debug("No retraining triggers detected")

def run_monitoring_loop():
    """Run the monitoring loop to check for retraining triggers"""
    logger.info("Starting auto-retraining monitoring loop")
    
    # Schedule retraining job
    schedule_retraining_job()
    
    # Initial check
    check_for_retraining_triggers()
    
    # Run loop
    try:
        while True:
            schedule.run_pending()
            
            # Check for triggers every hour
            check_for_retraining_triggers()
            
            # Wait before next check
            time.sleep(3600)  # Check every hour
            
    except KeyboardInterrupt:
        logger.info("Monitoring loop stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
        send_notification(f"Auto-retraining monitoring loop failed: {str(e)}", level="error")

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Auto-retraining module for object detection models")
    parser.add_argument("--retrain-now", action="store_true", help="Trigger immediate retraining")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring loop")
    parser.add_argument("--check", action="store_true", help="Check if retraining is needed")
    args = parser.parse_args()
    
    if args.retrain_now:
        # Trigger immediate retraining
        retrain_model(trigger_reason="manual")
    elif args.check:
        # Just check if retraining is needed
        needed, reason = check_retraining_needed()
        if needed:
            print(f"Retraining is needed. Reason: {reason}")
        else:
            print("No retraining needed at this time.")
    elif args.monitor:
        # Start monitoring loop
        run_monitoring_loop()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()