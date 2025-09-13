import os
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

from tqdm import tqdm


def setup_optimizer_and_scheduler(model, config):
    """Setup optimizer and scheduler"""
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    if config["training"]["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config["training"]["scheduler_patience"],
            factor=0.5,
            verbose=True,
        )
    elif config["training"]["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch in dataloader:
        categorical = batch["categorical"].to(device)
        continuous = batch["continuous"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(categorical, continuous)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def validate_epoch(
    model, dataloader, criterion, device, metrics_to_calculate=["mae", "mape"]
):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            categorical = batch["categorical"].to(device)
            continuous = batch["continuous"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(categorical, continuous)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate requested metrics
    metrics = {
        "loss": total_loss / len(dataloader),
    }

    if "mae" in metrics_to_calculate:
        metrics["mae"] = mean_absolute_error(all_targets, all_predictions)

    if "mape" in metrics_to_calculate:
        # Avoid division by zero
        mask = all_targets != 0
        if mask.sum() > 0:
            metrics["mape"] = (
                np.mean(
                    np.abs(
                        (all_targets[mask] - all_predictions[mask]) / all_targets[mask]
                    )
                )
                * 100
            )
        else:
            metrics["mape"] = float("inf")

    if "rmse" in metrics_to_calculate:
        mse = mean_squared_error(all_targets, all_predictions)
        metrics["rmse"] = np.sqrt(mse)

    if "mse" in metrics_to_calculate:
        metrics["mse"] = mean_squared_error(all_targets, all_predictions)

    return metrics


def train_model(model, train_loader, val_loader, config, device):
    """Main training loop with enhanced progress tracking"""

    # Setup logging
    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Setup training components
    criterion = nn.MSELoss()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)

    # Training variables
    best_val_metric = float("inf")
    patience_counter = 0
    early_stopping_patience = config["training"]["early_stopping_patience"]

    model.to(device)

    # Create main progress bar for epochs
    epoch_pbar = tqdm(
        range(config["training"]["num_epochs"]),
        desc="Training Progress",
        unit="epoch",
        leave=True,
    )

    for epoch in epoch_pbar:
        # Training with progress bar
        train_loss = train_epoch_with_progress(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validation
        val_metrics = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            metrics_to_calculate=[config["validation"]["metric"], "mape"],
        )

        # Get the metric for early stopping
        val_metric = val_metrics[config["validation"]["metric"]]

        # Update progress bar with current metrics
        epoch_pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val MAE": f'{val_metrics["mae"]:.4f}',
                "Val MAPE": f'{val_metrics.get("mape", 0):.2f}%',
                "Best": f"{best_val_metric:.4f}",
                "Patience": f"{patience_counter}/{early_stopping_patience}",
            }
        )

        # Scheduler step
        if scheduler and config["training"]["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_metric)
        elif scheduler:
            scheduler.step()

        # Detailed logging at intervals
        if (epoch + 1) % config["logging"]["log_interval"] == 0:
            log_msg = f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
            log_msg += f"Train Loss: {train_loss:.4f}, "
            log_msg += f"Val MAE: {val_metrics['mae']:.4f}, "
            if "mape" in val_metrics:
                log_msg += f"Val MAPE: {val_metrics['mape']:.4f}"

            # Print detailed log (tqdm will handle positioning)
            tqdm.write(log_msg)
            logging.info(log_msg)

        # Early stopping and model saving
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                    "config": config,
                },
                os.path.join(config["exp_dir"], "best_model.pth"),
            )

            # Show improvement message
            tqdm.write(
                f"New best model saved! Val {config['validation']['metric']}: {best_val_metric:.4f}"
            )

        else:
            patience_counter += 1

        # Regular model saving
        if (epoch + 1) % config["logging"]["save_model_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric": val_metric,
                    "config": config,
                },
                os.path.join(config["exp_dir"], f"model_epoch_{epoch+1}.pth"),
            )
            tqdm.write(f"Model checkpoint saved: model_epoch_{epoch+1}.pth")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            tqdm.write(f"Early stopping triggered after {epoch+1} epochs")
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    epoch_pbar.close()
    return model


def train_epoch_with_progress(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch with progress bar"""
    model.train()
    total_loss = 0

    # Create progress bar for batches within epoch
    batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

    for batch_idx, batch in enumerate(batch_pbar):
        # Extract data based on your dataset structure
        continuous_data = batch["continuous"].to(device)
        categorical_data = batch["categorical"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            categorical_data,
            continuous_data,
        )

        # Calculate loss (masked)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update batch progress bar
        batch_pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
            }
        )

    batch_pbar.close()
    return total_loss / len(dataloader)


def evaluate_model(model, test_loader, device, metrics=["mae", "mape"]):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            categorical = batch["categorical"].to(device)
            continuous = batch["continuous"].to(device)
            targets = batch["target"].to(device)

            outputs = model(categorical, continuous).squeeze()

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate comprehensive metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    result_metrics = {}

    if "mae" in metrics:
        result_metrics["MAE"] = mean_absolute_error(all_targets, all_predictions)

    if "mape" in metrics:
        # Avoid division by zero
        mask = all_targets != 0
        if mask.sum() > 0:
            result_metrics["MAPE"] = (
                np.mean(
                    np.abs(
                        (all_targets[mask] - all_predictions[mask]) / all_targets[mask]
                    )
                )
                * 100
            )
        else:
            result_metrics["MAPE"] = float("inf")

    if "mse" in metrics:
        result_metrics["MSE"] = mean_squared_error(all_targets, all_predictions)

    if "rmse" in metrics:
        mse = mean_squared_error(all_targets, all_predictions)
        result_metrics["RMSE"] = np.sqrt(mse)

    return result_metrics, all_predictions, all_targets
