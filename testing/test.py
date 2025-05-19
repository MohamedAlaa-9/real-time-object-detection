import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import gc

class TestConfig:
    # Configuration (optimized for memory)
    BASE_DIR = Path("/home/sci/WSL_Space/real-time-object-detection")
    TEST_VIDEO = BASE_DIR / "uploads/test_03.mp4"
    MODELS_DIR = BASE_DIR / "ml_models" / "models"
    MODELS = {
        "Base": MODELS_DIR / "yolo11n.pt",
        "50ep": MODELS_DIR / "old_versions/old_best.pt", 
        "20ep": MODELS_DIR / "old_versions/best.pt"
    }
    OUTPUT_DIR = BASE_DIR / "testing/test_results"
    IMG_SIZE = 640
    CONF_THRESH = 0.5
    IOU_THRESH = 0.45
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_FRAMES = 20000  # Reduced from 300 to save memory
    SAMPLE_FRAMES = 5  # Reduced from 5 to save memory

def save_plot(data, metric, output_path):
    """Save individual metric plot"""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(data.index, data[metric], color='skyblue')
    plt.title(f"{metric} Comparison", fontsize=14)
    plt.ylabel(metric)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def generate_comparison_plot(results_df, improvement_df, output_path):
    """Generate combined summary plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Performance Summary
    metrics = ["FPS", "Avg Confidence", "Avg Detections"]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    x = np.arange(len(results_df.index))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i])
    
    ax1.set_title('Model Performance Summary', pad=20)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(results_df.index)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Improvement Comparison
    improvement_df.plot.bar(x='Comparison', ax=ax2, color=['#FF9E44', '#6D904F', '#E24A33'])
    ax2.set_title('Relative Improvement (%)', pad=20)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def main():
    config = TestConfig()
    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Verify video exists
    if not config.TEST_VIDEO.exists():
        print(f"❌ Video not found: {config.TEST_VIDEO}")
        return

    # Evaluate models one-by-one to save memory
    results = {}
    for name, path in config.MODELS.items():
        try:
            print(f"\n🔍 Evaluating {name}...")
            
            # Load model
            model = YOLO(str(path)).to(config.DEVICE)
            model.fuse()
            
            # Process video (memory-optimized)
            cap = cv2.VideoCapture(str(config.TEST_VIDEO))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_times = []
            detections = []
            confidences = []
            
            for _ in tqdm(range(min(frame_count, config.MAX_FRAMES)), desc="Processing"):
                ret, frame = cap.read()
                if not ret: break
                
                # Time inference
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    preds = model(frame, imgsz=config.IMG_SIZE, 
                                conf=config.CONF_THRESH, 
                                iou=config.IOU_THRESH,
                                verbose=False)[0]
                end.record()
                torch.cuda.synchronize()
                
                # Store results
                frame_times.append(start.elapsed_time(end))
                detections.append(len(preds.boxes) if preds.boxes else 0)
                confidences.append(preds.boxes.conf.mean().item() if preds.boxes else 0)
                
                # Clean up
                del frame, preds
                torch.cuda.empty_cache()
            
            cap.release()
            
            # Save results
            results[name] = {
                "FPS": 1000 / np.mean(frame_times),
                "Avg Detections": np.mean(detections),
                "Avg Confidence": np.mean(confidences)
            }
            
            # Clean up model
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error with {name}: {str(e)}")
    
    # Generate results DataFrame
    results_df = pd.DataFrame(results).T
    
    # Generate improvement report
    improvement_data = []
    models = list(results.keys())
    for i in range(1, len(models)):
        improvement_data.append({
            "Comparison": f"{models[i]} vs {models[i-1]}",
            "FPS": calculate_improvement(results[models[i]]["FPS"], results[models[i-1]]["FPS"]),
            "Confidence": calculate_improvement(results[models[i]]["Avg Confidence"], results[models[i-1]]["Avg Confidence"]),
            "Detections": calculate_improvement(results[models[i]]["Avg Detections"], results[models[i-1]]["Avg Detections"])
        })
    improvement_df = pd.DataFrame(improvement_data)
    
    # Generate all plots
    print("\n📊 Generating visualizations...")
    for metric in ["FPS", "Avg Confidence", "Avg Detections"]:
        save_plot(results_df, metric, config.OUTPUT_DIR / f"{metric.lower().replace(' ', '_')}.png")
    
    generate_comparison_plot(results_df, improvement_df, config.OUTPUT_DIR / "summary_comparison.png")
    
    # Save results
    results_df.to_csv(config.OUTPUT_DIR / "results.csv")
    improvement_df.to_csv(config.OUTPUT_DIR / "improvements.csv")
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {config.OUTPUT_DIR}")
    print("\n🏆 Final Results:")
    print(results_df)
    print("\n📈 Improvement Comparison:")
    print(improvement_df)

def calculate_improvement(new, old):
    return ((new - old) / old) * 100 if old != 0 else 0

if __name__ == "__main__":
    main()