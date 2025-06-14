#!/usr/bin/env python3
"""
Enhanced training script with comprehensive MLflow tracking including:
- System metrics (CPU, memory, GPU)
- Artifact logging (model, plots, data)
- Autologging integration
- Performance monitoring
"""

import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Try to import GPU monitoring
try:
    import nvidia_ml_py as nvml

    NVIDIA_AVAILABLE = True
    nvml.nvmlInit()
except (ImportError, Exception):
    NVIDIA_AVAILABLE = False
    print("⚠️  NVIDIA GPU monitoring not available")

try:
    import gpustat

    GPUSTAT_AVAILABLE = True
except ImportError:
    GPUSTAT_AVAILABLE = False


class SystemMetricsLogger:
    """Log system metrics during training"""

    def __init__(self):
        self.metrics_history = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_used_gb": [],
            "memory_available_gb": [],
            "disk_usage_percent": [],
            "gpu_metrics": [],
        }

        # Get system info
        self.system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version(),
        }

        if NVIDIA_AVAILABLE:
            try:
                self.gpu_count = nvml.nvmlDeviceGetCount()
                self.system_info["gpu_count"] = self.gpu_count

                # Get GPU info
                gpu_info = []
                for i in range(self.gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.append(
                        {"name": name, "memory_total_mb": memory_info.total / (1024**2)}
                    )
                self.system_info["gpu_info"] = gpu_info
            except Exception as e:
                print(f"⚠️  GPU info collection failed: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0

    def collect_metrics(self):
        """Collect current system metrics"""
        timestamp = time.time()

        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        self.metrics_history["timestamp"].append(timestamp)
        self.metrics_history["cpu_percent"].append(cpu_percent)
        self.metrics_history["memory_percent"].append(memory.percent)
        self.metrics_history["memory_used_gb"].append(memory.used / (1024**3))
        self.metrics_history["memory_available_gb"].append(memory.available / (1024**3))
        self.metrics_history["disk_usage_percent"].append(disk.percent)

        # GPU metrics
        gpu_metrics = []
        if NVIDIA_AVAILABLE and self.gpu_count > 0:
            try:
                for i in range(self.gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)

                    # GPU utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)

                    # Memory info
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_percent = (memory_info.used / memory_info.total) * 100

                    # Temperature
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(
                            handle, nvml.NVML_TEMPERATURE_GPU
                        )
                    except:
                        temp = 0

                    gpu_metrics.append(
                        {
                            "gpu_id": i,
                            "gpu_utilization": util.gpu,
                            "memory_utilization": util.memory,
                            "memory_used_percent": memory_used_percent,
                            "memory_used_mb": memory_info.used / (1024**2),
                            "memory_total_mb": memory_info.total / (1024**2),
                            "temperature": temp,
                        }
                    )
            except Exception as e:
                print(f"⚠️  GPU metrics collection failed: {e}")

        self.metrics_history["gpu_metrics"].append(gpu_metrics)

        return {
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": disk.percent,
            "gpu_metrics": gpu_metrics,
        }

    def log_to_mlflow(self, step=None):
        """Log current metrics to MLflow"""
        current_metrics = self.collect_metrics()

        # Log basic system metrics
        mlflow.log_metric(
            "system/cpu_percent", current_metrics["cpu_percent"], step=step
        )
        mlflow.log_metric(
            "system/memory_percent", current_metrics["memory_percent"], step=step
        )
        mlflow.log_metric(
            "system/memory_used_gb", current_metrics["memory_used_gb"], step=step
        )
        mlflow.log_metric(
            "system/disk_usage_percent",
            current_metrics["disk_usage_percent"],
            step=step,
        )

        # Log GPU metrics
        for gpu_data in current_metrics["gpu_metrics"]:
            gpu_id = gpu_data["gpu_id"]
            mlflow.log_metric(
                f"system/gpu_{gpu_id}_utilization",
                gpu_data["gpu_utilization"],
                step=step,
            )
            mlflow.log_metric(
                f"system/gpu_{gpu_id}_memory_percent",
                gpu_data["memory_used_percent"],
                step=step,
            )
            mlflow.log_metric(
                f"system/gpu_{gpu_id}_memory_used_mb",
                gpu_data["memory_used_mb"],
                step=step,
            )
            mlflow.log_metric(
                f"system/gpu_{gpu_id}_temperature", gpu_data["temperature"], step=step
            )

    def create_system_metrics_plots(self):
        """Create plots of system metrics over time"""
        if len(self.metrics_history["timestamp"]) < 2:
            return []

        plots = []

        # Convert timestamps to relative time in minutes
        start_time = self.metrics_history["timestamp"][0]
        time_minutes = [
            (t - start_time) / 60 for t in self.metrics_history["timestamp"]
        ]

        # CPU and Memory plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # CPU usage
        ax1.plot(time_minutes, self.metrics_history["cpu_percent"], "b-", label="CPU %")
        ax1.set_ylabel("CPU Usage (%)")
        ax1.set_title("System CPU Usage Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Memory usage
        ax2.plot(
            time_minutes, self.metrics_history["memory_percent"], "r-", label="Memory %"
        )
        ax2.plot(
            time_minutes,
            self.metrics_history["memory_used_gb"],
            "g-",
            label="Memory Used (GB)",
        )
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Memory Usage")
        ax2.set_title("System Memory Usage Over Time")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plot_path = tempfile.mktemp(suffix="_system_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots.append(plot_path)

        # GPU metrics plot (if available)
        if self.gpu_count > 0 and len(self.metrics_history["gpu_metrics"]) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            for i in range(min(self.gpu_count, 2)):  # Plot first 2 GPUs
                if len(axes.shape) == 1:
                    ax_util = axes[0] if i == 0 else None
                    ax_mem = axes[1] if i == 0 else None
                else:
                    ax_util = axes[0, i]
                    ax_mem = axes[1, i]

                if ax_util is None or ax_mem is None:
                    continue

                # Extract GPU data for this GPU
                gpu_util = []
                gpu_mem = []
                gpu_temp = []

                for gpu_data_list in self.metrics_history["gpu_metrics"]:
                    gpu_data = next(
                        (g for g in gpu_data_list if g["gpu_id"] == i), None
                    )
                    if gpu_data:
                        gpu_util.append(gpu_data["gpu_utilization"])
                        gpu_mem.append(gpu_data["memory_used_percent"])
                        gpu_temp.append(gpu_data["temperature"])
                    else:
                        gpu_util.append(0)
                        gpu_mem.append(0)
                        gpu_temp.append(0)

                # GPU utilization and temperature
                ax_util_twin = ax_util.twinx()
                line1 = ax_util.plot(
                    time_minutes, gpu_util, "b-", label=f"GPU {i} Utilization %"
                )
                line2 = ax_util_twin.plot(
                    time_minutes, gpu_temp, "r--", label=f"GPU {i} Temp (°C)"
                )

                ax_util.set_ylabel("GPU Utilization (%)", color="b")
                ax_util_twin.set_ylabel("Temperature (°C)", color="r")
                ax_util.set_title(f"GPU {i} Utilization and Temperature")
                ax_util.grid(True, alpha=0.3)

                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax_util.legend(lines, labels, loc="upper left")

                # GPU memory
                ax_mem.plot(time_minutes, gpu_mem, "g-", label=f"GPU {i} Memory %")
                ax_mem.set_ylabel("GPU Memory Usage (%)")
                ax_mem.set_xlabel("Time (minutes)")
                ax_mem.set_title(f"GPU {i} Memory Usage")
                ax_mem.grid(True, alpha=0.3)
                ax_mem.legend()

            plt.tight_layout()

            # Save GPU plot
            gpu_plot_path = tempfile.mktemp(suffix="_gpu_metrics.png")
            plt.savefig(gpu_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots.append(gpu_plot_path)

        return plots


class EnhancedTrainingDemo:
    """Enhanced training demo with comprehensive MLflow integration"""

    def __init__(self):
        self.system_logger = SystemMetricsLogger()
        self.training_history = []

    def create_demo_model_and_data(self):
        """Create demo PyTorch model and data for training demonstration"""

        # Simple neural network
        class DemoModel(nn.Module):
            def __init__(self, input_size=784, hidden_size=128, num_classes=10):
                super(DemoModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = x.view(-1, 784)  # Flatten
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        # Create model
        model = DemoModel()

        # Generate demo data (simulating MNIST-like data)
        np.random.seed(42)
        torch.manual_seed(42)

        # Training data
        X_train = torch.randn(1000, 1, 28, 28)  # 1000 samples, 1 channel, 28x28
        y_train = torch.randint(0, 10, (1000,))  # 10 classes

        # Test data
        X_test = torch.randn(200, 1, 28, 28)
        y_test = torch.randint(0, 10, (200,))

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return model, train_loader, test_loader

    def train_with_comprehensive_logging(self):
        """Train model with comprehensive MLflow logging"""

        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5001")

        # Set experiment
        experiment_name = "Enhanced-Training-With-System-Metrics"
        mlflow.set_experiment(experiment_name)

        print(f"🚀 Starting enhanced training with comprehensive logging")
        print(f"📊 Experiment: {experiment_name}")
        print(f"🔗 MLflow UI: http://localhost:5001")

        with mlflow.start_run(
            run_name=f"enhanced-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):

            # Enable MLflow autolog for PyTorch
            mlflow.pytorch.autolog(log_models=True, log_datasets=True)

            # Log system information
            mlflow.log_params(self.system_logger.system_info)

            # Create model and data
            model, train_loader, test_loader = self.create_demo_model_and_data()

            # Training configuration
            config = {
                "learning_rate": 0.001,
                "epochs": 5,
                "batch_size": 32,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "model_size_mb": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                / (1024**2),
            }

            # Log configuration
            mlflow.log_params(config)

            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

            print(f"\n📋 Training Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")

            print(f"\n🖥️  System Information:")
            for key, value in self.system_logger.system_info.items():
                if key != "gpu_info":
                    print(f"  {key}: {value}")

            if "gpu_info" in self.system_logger.system_info:
                print(f"  GPU Info:")
                for i, gpu in enumerate(self.system_logger.system_info["gpu_info"]):
                    print(
                        f"    GPU {i}: {gpu['name']} ({gpu['memory_total_mb']:.0f} MB)"
                    )

            # Training loop with system metrics logging
            print(f"\n🏋️  Training Progress:")

            for epoch in range(config["epochs"]):
                model.train()
                epoch_loss = 0
                correct = 0
                total = 0

                # Log system metrics at start of epoch
                self.system_logger.log_to_mlflow(step=epoch)

                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                    # Log batch metrics periodically
                    if batch_idx % 10 == 0:
                        batch_step = epoch * len(train_loader) + batch_idx
                        mlflow.log_metric("batch_loss", loss.item(), step=batch_step)

                        # Log system metrics every few batches
                        if batch_idx % 5 == 0:
                            self.system_logger.log_to_mlflow(step=batch_step)

                # Calculate epoch metrics
                avg_loss = epoch_loss / len(train_loader)
                accuracy = 100 * correct / total

                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for data, target in test_loader:
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()

                val_avg_loss = val_loss / len(test_loader)
                val_accuracy = 100 * val_correct / val_total

                # Log epoch metrics
                epoch_metrics = {
                    "train_loss": avg_loss,
                    "train_accuracy": accuracy,
                    "val_loss": val_avg_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": config["learning_rate"],
                    "epoch": epoch + 1,
                }

                mlflow.log_metrics(epoch_metrics, step=epoch)
                self.training_history.append(epoch_metrics)

                print(
                    f"  Epoch {epoch+1}/{config['epochs']}: "
                    f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%, "
                    f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
                )

            # Create and log training plots
            self.create_and_log_training_plots()

            # Create and log system metrics plots
            system_plots = self.system_logger.create_system_metrics_plots()
            for plot_path in system_plots:
                plot_name = Path(plot_path).stem
                mlflow.log_artifact(plot_path, artifact_path="system_metrics")
                os.remove(plot_path)  # Clean up temp file

            # Log final model
            mlflow.pytorch.log_model(model, "final_model")

            # Create and log model summary
            model_summary = {
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model_size_mb": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                / (1024**2),
                "final_train_accuracy": self.training_history[-1]["train_accuracy"],
                "final_val_accuracy": self.training_history[-1]["val_accuracy"],
                "best_val_accuracy": max(
                    h["val_accuracy"] for h in self.training_history
                ),
                "training_time_minutes": len(
                    self.system_logger.metrics_history["timestamp"]
                )
                * 10
                / 60,  # Approximate
            }

            # Save model summary as artifact
            summary_path = tempfile.mktemp(suffix="_model_summary.json")
            with open(summary_path, "w") as f:
                json.dump(model_summary, f, indent=2)
            mlflow.log_artifact(summary_path, artifact_path="model_info")
            os.remove(summary_path)

            # Log final metrics
            mlflow.log_metrics(model_summary)

            print(f"\n✅ Training completed successfully!")
            print(f"\n📊 Final Results:")
            print(
                f"  Best Validation Accuracy: {model_summary['best_val_accuracy']:.2f}%"
            )
            print(
                f"  Final Training Accuracy: {model_summary['final_train_accuracy']:.2f}%"
            )
            print(f"  Model Parameters: {model_summary['total_parameters']:,}")
            print(f"  Model Size: {model_summary['model_size_mb']:.2f} MB")

            # Get current run info
            run = mlflow.active_run()
            run_id = run.info.run_id

            print(f"\n🔗 View Results:")
            print(
                f"  MLflow UI: http://localhost:5001/#/experiments/{mlflow.get_experiment_by_name(experiment_name).experiment_id}/runs/{run_id}"
            )
            print(f"  Run ID: {run_id}")

            return run_id

    def create_and_log_training_plots(self):
        """Create and log training visualization plots"""
        if not self.training_history:
            return

        # Extract metrics
        epochs = [h["epoch"] for h in self.training_history]
        train_losses = [h["train_loss"] for h in self.training_history]
        val_losses = [h["val_loss"] for h in self.training_history]
        train_accs = [h["train_accuracy"] for h in self.training_history]
        val_accs = [h["val_accuracy"] for h in self.training_history]

        # Create training plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss plots
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", marker="o")
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plots
        ax2.plot(epochs, train_accs, "b-", label="Training Accuracy", marker="o")
        ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        ax3.plot(epochs, loss_diff, "g-", label="|Train Loss - Val Loss|", marker="^")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss Difference")
        ax3.set_title("Training-Validation Loss Gap")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Learning curve summary
        ax4.bar(
            ["Train Acc", "Val Acc"],
            [train_accs[-1], val_accs[-1]],
            color=["blue", "red"],
            alpha=0.7,
        )
        ax4.set_ylabel("Final Accuracy (%)")
        ax4.set_title("Final Model Performance")
        ax4.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (label, value) in enumerate(
            zip(["Train Acc", "Val Acc"], [train_accs[-1], val_accs[-1]])
        ):
            ax4.text(i, value + 0.5, f"{value:.1f}%", ha="center", va="bottom")

        plt.tight_layout()

        # Save and log plot
        plot_path = tempfile.mktemp(suffix="_training_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(plot_path, artifact_path="training_plots")
        plt.close()
        os.remove(plot_path)  # Clean up temp file


def main():
    """Main function to run enhanced training demonstration"""

    print("🔧 Enhanced MLflow Training with System Metrics & Artifacts")
    print("=" * 60)

    # Check MLflow connectivity
    try:
        import requests

        response = requests.get("http://localhost:5001/health", timeout=5)
        print("✅ MLflow server is accessible")
    except Exception as e:
        print(f"❌ MLflow server not accessible: {e}")
        print("Please ensure MLflow is running at http://localhost:5001")
        return

    # Create and run training
    trainer = EnhancedTrainingDemo()
    run_id = trainer.train_with_comprehensive_logging()

    print(f"\n🎉 Enhanced training demonstration completed!")
    print(f"\n🚀 What was logged to MLflow:")
    print(f"  ✅ System metrics (CPU, memory, GPU)")
    print(f"  ✅ Training metrics (loss, accuracy)")
    print(f"  ✅ Model artifacts (PyTorch model)")
    print(f"  ✅ Training plots and visualizations")
    print(f"  ✅ System performance plots")
    print(f"  ✅ Model summary and metadata")
    print(f"  ✅ Configuration parameters")

    print(f"\n📋 To enable this in your training:")
    print(f"  1. Add system monitoring dependencies")
    print(f"  2. Use mlflow.pytorch.autolog()")
    print(f"  3. Log system metrics during training")
    print(f"  4. Create and log visualization artifacts")
    print(f"  5. Enable MLflow system metrics environment variables")


if __name__ == "__main__":
    main()
