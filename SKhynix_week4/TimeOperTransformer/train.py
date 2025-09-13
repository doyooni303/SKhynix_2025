#!/usr/bin/env python3
"""
훈련 및 평가 함수들
Loss 함수, 메트릭 계산, 훈련 루프, 테스트 등
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from utils import logger


class MaskedMSELoss(nn.Module):
    """마스크를 고려한 MSE Loss"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, masks):
        valid_mask = ~masks

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]

        mse_loss = nn.functional.mse_loss(
            valid_predictions, valid_targets, reduction=self.reduction
        )
        return mse_loss


def compute_metrics(predictions, targets, masks, epsilon=1e-8):
    """성능 메트릭 계산 (MSE, RMSE, MAE, MAPE)"""
    valid_mask = ~masks
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    if len(valid_predictions) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}

    mse = torch.mean((valid_predictions - valid_targets) ** 2).item()
    rmse = torch.sqrt(torch.mean((valid_predictions - valid_targets) ** 2)).item()
    mae = torch.mean(torch.abs(valid_predictions - valid_targets)).item()

    abs_targets = torch.abs(valid_targets)
    abs_errors = torch.abs(valid_predictions - valid_targets)
    safe_targets = torch.clamp(abs_targets, min=epsilon)
    mape = torch.mean(abs_errors / safe_targets * 100).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "valid_count": len(valid_predictions),
    }


def train_epoch(
    model, train_dataloader, criterion, optimizer, device, epoch, log_frequency=50
):
    """한 에폭 훈련"""
    model.train()
    total_loss = 0.0
    total_metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}

    # tqdm 진행바 설정
    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for batch_idx, batch in pbar:
        continuous_data = batch["continuous_data"].to(device)
        categorical_data = batch["categorical_data"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        time_masks = batch["time_masks"].to(device)

        predictions = model(continuous_data, categorical_data, masks, time_masks)
        loss = criterion(predictions, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_metrics = compute_metrics(predictions, targets, masks)

        total_loss += loss.item()
        for key in ["mse", "rmse", "mae", "mape"]:
            total_metrics[key] += batch_metrics[key]
        total_metrics["valid_count"] += batch_metrics["valid_count"]

        # tqdm 진행바 업데이트 (실시간 메트릭 표시)
        current_avg_loss = total_loss / (batch_idx + 1)
        current_avg_mape = total_metrics["mape"] / (batch_idx + 1)
        pbar.set_postfix(
            {
                "Loss": f"{current_avg_loss:.4f}",
                "MAPE": f"{current_avg_mape:.2f}%",
                "Batch_Loss": f"{loss.item():.4f}",
            }
        )

        # 주기적 로깅 (선택적)
        if batch_idx % log_frequency == 0 and batch_idx > 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{len(train_dataloader)}, "
                f'Loss: {loss.item():.4f}, MAPE: {batch_metrics["mape"]:.2f}%'
            )

    pbar.close()

    avg_loss = total_loss / len(train_dataloader)
    for key in ["mse", "rmse", "mae", "mape"]:
        total_metrics[key] = total_metrics[key] / len(train_dataloader)

    return avg_loss, total_metrics


def validate_epoch(model, val_dataloader, criterion, device, epoch=None):
    """검증 에폭"""
    model.eval()
    total_loss = 0.0
    total_metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}

    # tqdm 진행바 설정
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(
        val_dataloader,
        desc=desc,
        leave=False,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            continuous_data = batch["continuous_data"].to(device)
            categorical_data = batch["categorical_data"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)
            time_masks = batch["time_masks"].to(device)

            predictions = model(continuous_data, categorical_data, masks, time_masks)
            loss = criterion(predictions, targets, masks)
            batch_metrics = compute_metrics(predictions, targets, masks)

            total_loss += loss.item()
            for key in ["mse", "rmse", "mae", "mape"]:
                total_metrics[key] += batch_metrics[key]
            total_metrics["valid_count"] += batch_metrics["valid_count"]

            # tqdm 진행바 업데이트
            current_avg_loss = total_loss / (batch_idx + 1)
            current_avg_mape = total_metrics["mape"] / (batch_idx + 1)
            pbar.set_postfix(
                {"Loss": f"{current_avg_loss:.4f}", "MAPE": f"{current_avg_mape:.2f}%"}
            )

    pbar.close()

    avg_loss = total_loss / len(val_dataloader)
    for key in ["mse", "rmse", "mae", "mape"]:
        total_metrics[key] = total_metrics[key] / len(val_dataloader)

    return avg_loss, total_metrics


def train_model(
    model, train_dataloader, val_dataloader, training_config: dict, device: str
):
    """모델 훈련 메인 함수"""
    num_epochs = training_config.get("num_epochs", 100)
    learning_rate = training_config.get("learning_rate", 1e-3)
    patience = training_config.get("patience", 10)
    save_path = training_config.get("save_path", "best_model.pth")
    log_frequency = training_config.get("log_frequency", 50)

    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True
    )

    model = model.to(device)

    train_history = {"loss": [], "mse": [], "rmse": [], "mae": [], "mape": []}
    val_history = {"loss": [], "mse": [], "rmse": [], "mae": [], "mape": []}

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"모델 훈련 시작: {num_epochs}개 에폭, 학습률 {learning_rate}")

    # 전체 에폭에 대한 tqdm 진행바
    epoch_pbar = tqdm(
        range(1, num_epochs + 1),
        desc="Training Progress",
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for epoch in epoch_pbar:
        train_loss, train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch, log_frequency
        )
        val_loss, val_metrics = validate_epoch(
            model, val_dataloader, criterion, device, epoch
        )

        scheduler.step(val_loss)

        train_history["loss"].append(train_loss)
        for key in ["mse", "rmse", "mae", "mape"]:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        val_history["loss"].append(val_loss)

        # 에폭별 결과 로깅
        logger.info(
            f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f'Train MAPE={train_metrics["mape"]:.2f}%, Val MAPE={val_metrics["mape"]:.2f}%'
        )

        # 전체 에폭 진행바 업데이트
        epoch_pbar.set_postfix(
            {
                "T_Loss": f"{train_loss:.4f}",
                "V_Loss": f"{val_loss:.4f}",
                "T_MAPE": f'{train_metrics["mape"]:.2f}%',
                "V_MAPE": f'{val_metrics["mape"]:.2f}%',
                "Best": f"{best_val_loss:.4f}",
                "Patience": f"{patience_counter}/{patience}",
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                },
                save_path,
            )
            logger.info(f"  → Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            epoch_pbar.close()
            break

    if patience_counter < patience:
        epoch_pbar.close()

    return {
        "train_history": train_history,
        "val_history": val_history,
        "best_val_loss": best_val_loss,
    }


def test_model_with_structure(model, test_dataloader, device, model_path):
    """구조화된 테스트 (timekey_hr, oper_id 복원)"""
    logger.info(f"저장된 모델 로드: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    criterion = MaskedMSELoss()
    structured_predictions = []
    valid_predictions_list = []
    valid_targets_list = []
    total_loss = 0.0

    logger.info(f"구조화된 테스트 시작")

    # 테스트용 tqdm 진행바
    pbar = tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc="Testing",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    with torch.no_grad():
        for batch_idx, batch in pbar:
            continuous_data = batch["continuous_data"].to(device)
            categorical_data = batch["categorical_data"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)
            time_masks = batch["time_masks"].to(device)

            timekey_hr_info_batch = batch["timekey_hr_info"]
            oper_id_info_batch = batch["oper_id_info"]

            predictions = model(continuous_data, categorical_data, masks, time_masks)
            loss = criterion(predictions, targets, masks)
            total_loss += loss.item()

            predictions_cpu = predictions.cpu()
            targets_cpu = targets.cpu()
            masks_cpu = masks.cpu()

            batch_size = predictions_cpu.shape[0]
            for sample_idx in range(batch_size):
                sample_predictions = predictions_cpu[sample_idx]
                sample_targets = targets_cpu[sample_idx]
                sample_masks = masks_cpu[sample_idx]

                sample_timekey_hr_info = timekey_hr_info_batch[sample_idx]
                sample_oper_id_info = oper_id_info_batch[sample_idx]

                for time_idx, (timekey_hr, oper_ids_array) in enumerate(
                    zip(sample_timekey_hr_info, sample_oper_id_info)
                ):
                    if timekey_hr is not None and len(oper_ids_array) > 0:
                        for oper_idx_in_array, actual_oper_id in enumerate(
                            oper_ids_array
                        ):
                            if not sample_masks[time_idx, oper_idx_in_array]:
                                pred_val = sample_predictions[
                                    time_idx, oper_idx_in_array
                                ].item()
                                target_val = sample_targets[
                                    time_idx, oper_idx_in_array
                                ].item()

                                structured_predictions.append(
                                    {
                                        "timekey_hr": timekey_hr,
                                        "oper_id": actual_oper_id,
                                        "predicted": pred_val,
                                        "actual": target_val,
                                    }
                                )

                                valid_predictions_list.append(pred_val)
                                valid_targets_list.append(target_val)

            # tqdm 진행바 업데이트
            current_avg_loss = total_loss / (batch_idx + 1)
            predictions_count = len(structured_predictions)
            pbar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Predictions": f"{predictions_count:,}",
                }
            )

    pbar.close()

    avg_loss = total_loss / len(test_dataloader)

    if len(valid_predictions_list) > 0:
        all_valid_predictions = torch.tensor(valid_predictions_list)
        all_valid_targets = torch.tensor(valid_targets_list)

        mse = torch.mean((all_valid_predictions - all_valid_targets) ** 2).item()
        rmse = torch.sqrt(
            torch.mean((all_valid_predictions - all_valid_targets) ** 2)
        ).item()
        mae = torch.mean(torch.abs(all_valid_predictions - all_valid_targets)).item()

        epsilon = 1e-8
        abs_targets = torch.abs(all_valid_targets)
        abs_errors = torch.abs(all_valid_predictions - all_valid_targets)
        safe_targets = torch.clamp(abs_targets, min=epsilon)
        mape = torch.mean(abs_errors / safe_targets * 100).item()

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "valid_count": len(valid_predictions_list),
        }
    else:
        metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}

    logger.info(
        f"테스트 결과: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%"
    )
    logger.info(f"구조화된 예측 결과: {len(structured_predictions):,}개")

    return {
        "test_loss": avg_loss,
        "metrics": metrics,
        "structured_predictions": structured_predictions,
    }
