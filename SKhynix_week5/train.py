import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)


class MaskedMSELoss(nn.Module):
    """패딩을 고려한 안전한 MSE Loss - NaN 방지"""

    def __init__(self, padding_value: float = -9999.0):
        super().__init__()
        self.padding_value = padding_value

    def forward(self, predictions, targets, masks):
        """
        Args:
            predictions: [batch_size, seq_len]
            targets: [batch_size, seq_len]
            masks: [batch_size, seq_len] (True = 패딩)
        """
        # 패딩되지 않은 위치만 선택
        valid_mask = ~masks

        if valid_mask.sum() == 0:
            print("⚠️ 경고: 유효한 데이터가 없습니다!")
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]

        # NaN/Inf 체크 및 제거
        finite_mask = torch.isfinite(valid_predictions) & torch.isfinite(valid_targets)

        if finite_mask.sum() == 0:
            print("⚠️ 경고: finite한 값이 없습니다!")
            return torch.tensor(
                1e6, device=predictions.device, requires_grad=True
            )  # 큰 손실값 반환

        valid_predictions = valid_predictions[finite_mask]
        valid_targets = valid_targets[finite_mask]

        # MSE 계산
        mse = F.mse_loss(valid_predictions, valid_targets)

        # NaN 체크
        if torch.isnan(mse) or torch.isinf(mse):
            print(f"⚠️ 경고: MSE가 {mse.item()}입니다!")
            return torch.tensor(1e6, device=predictions.device, requires_grad=True)

        return mse


def compute_metrics(predictions, targets, masks, padding_value: float = -9999.0):
    """패딩을 고려한 메트릭 계산"""
    valid_mask = ~masks

    if valid_mask.sum() == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}

    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    # CPU로 변환
    valid_predictions = valid_predictions.detach().cpu().numpy()
    valid_targets = valid_targets.detach().cpu().numpy()

    mse = np.mean((valid_predictions - valid_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(valid_predictions - valid_targets))

    # MAPE 계산 (0으로 나누기 방지)
    epsilon = 1e-8
    abs_targets = np.abs(valid_targets)
    abs_errors = np.abs(valid_predictions - valid_targets)
    safe_targets = np.maximum(abs_targets, epsilon)
    mape = np.mean(abs_errors / safe_targets * 100)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "valid_count": len(valid_predictions),
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """한 에폭 훈련 - 통합 임베딩 데이터셋용"""
    model.train()
    total_loss = 0.0
    total_metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch} [Train]",
        leave=False,
    )

    for batch_idx, batch in pbar:
        # 새로운 데이터셋 구조에 맞는 키 사용
        features = batch["features"].to(device)  # 이미 결합된 특성
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()

        # 모든 모델이 동일한 인터페이스 사용
        predictions = model(features, masks)
        loss = criterion(predictions, targets, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # 메트릭 계산
        with torch.no_grad():
            batch_metrics = compute_metrics(predictions, targets, masks)

        total_loss += loss.item()
        for key in ["mse", "rmse", "mae", "mape"]:
            total_metrics[key] += batch_metrics[key]
        total_metrics["valid_count"] += batch_metrics["valid_count"]

        # 진행바 업데이트
        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "MAPE": f"{batch_metrics['mape']:.2f}%"}
        )

    pbar.close()

    # 평균 계산
    avg_loss = total_loss / len(dataloader)
    for key in ["mse", "rmse", "mae", "mape"]:
        total_metrics[key] = total_metrics[key] / len(dataloader)

    return avg_loss, total_metrics


def validate_epoch(model, dataloader, criterion, device, epoch=None):
    """검증 에폭 - 통합 임베딩 데이터셋용"""
    model.eval()
    total_loss = 0
    total_metrics = {"mse": 0, "rmse": 0, "mae": 0, "mape": 0, "valid_count": 0}

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in pbar:
            # 새로운 데이터셋 구조에 맞는 키 사용
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)

            # 모든 모델이 동일한 인터페이스 사용
            predictions = model(features, masks)
            loss = criterion(predictions, targets, masks)

            batch_metrics = compute_metrics(predictions, targets, masks)

            total_loss += loss.item()
            for key in ["mse", "rmse", "mae", "mape"]:
                total_metrics[key] += batch_metrics[key]
            total_metrics["valid_count"] += batch_metrics["valid_count"]

            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "MAPE": f"{batch_metrics['mape']:.2f}%"}
            )

    pbar.close()

    avg_loss = total_loss / len(dataloader)
    for key in ["mse", "rmse", "mae", "mape"]:
        total_metrics[key] = total_metrics[key] / len(dataloader)

    return avg_loss, total_metrics


def train_model(model, train_loader, val_loader, training_config, device, save_path):
    """메인 훈련 루프 - 통합 임베딩 데이터셋용 (검증 간격 설정 가능)"""

    num_epochs = training_config.get("num_epochs", 100)
    learning_rate = training_config.get("learning_rate", 1e-3)
    patience = training_config.get("patience", 20)
    padding_value = training_config.get("padding_value", -9999.0)

    # 검증 간격 설정 (새로 추가)
    val_interval = training_config.get("val_interval", 1)  # 기본값: 매 에폭마다 검증
    log_interval = training_config.get("log_interval", 1)  # 기본값: 매 에폭마다 로깅

    # 손실 함수 및 옵티마이저
    criterion = MaskedMSELoss(padding_value)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True
    )

    model = model.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    last_val_loss = None
    last_val_metrics = None

    logger.info(f"훈련 시작: {num_epochs} 에폭, 학습률 {learning_rate}")
    logger.info(f"모델 타입: {training_config.get('model_type', 'unknown')}")
    logger.info(f"검증 간격: {val_interval} 에폭마다")
    logger.info(f"로깅 간격: {log_interval} 에폭마다")

    # 에폭 진행바
    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training Progress")

    for epoch in epoch_pbar:
        # 훈련은 매 에폭마다 실시
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 검증은 지정된 간격마다 또는 마지막 에폭에서 실시
        should_validate = (epoch % val_interval == 0) or (epoch == num_epochs)

        if should_validate:
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            last_val_loss = val_loss
            last_val_metrics = val_metrics

            # 스케줄러 업데이트 (검증이 실시된 경우만)
            scheduler.step(val_loss)
        else:
            # 검증하지 않는 에폭에서는 이전 검증 결과 사용
            val_loss = last_val_loss if last_val_loss is not None else float("inf")
            val_metrics = (
                last_val_metrics
                if last_val_metrics is not None
                else {
                    "mse": 0.0,
                    "rmse": 0.0,
                    "mae": 0.0,
                    "mape": 0.0,
                    "valid_count": 0,
                }
            )

        # 로깅 (지정된 간격마다 또는 검증이 실시된 경우)
        should_log = (
            (epoch % log_interval == 0) or should_validate or (epoch == num_epochs)
        )

        if should_log:
            if should_validate:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f'Train MAPE={train_metrics["mape"]:.2f}%, Val MAPE={val_metrics["mape"]:.2f}%'
                )
            else:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, "
                    f'Train MAPE={train_metrics["mape"]:.2f}% (검증 생략)'
                )

        # 진행바 업데이트
        val_status = "검증됨" if should_validate else "이전값"
        epoch_pbar.set_postfix(
            {
                "T_Loss": f"{train_loss:.4f}",
                "V_Loss": f"{val_loss:.4f}({val_status})",
                "V_MAPE": f'{val_metrics["mape"]:.2f}%',
                "Best": f"{best_val_loss:.4f}",
                "Patience": f"{patience_counter}/{patience}",
            }
        )

        # 최고 모델 저장 (검증이 실시된 경우만)
        if should_validate and val_loss < best_val_loss:
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
                    "model_type": training_config.get("model_type", "unknown"),
                    "feature_dim": training_config.get("feature_dim", None),
                    "config": training_config,
                },
                save_path,
            )

            logger.info(f"  → Best model saved! (Val Loss: {val_loss:.4f})")
        elif should_validate:
            # 검증은 했지만 성능이 개선되지 않은 경우
            patience_counter += 1
        # 검증하지 않은 경우 patience_counter는 증가시키지 않음

        # 조기 종료 (검증이 실시된 경우만 확인)
        if should_validate and patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    epoch_pbar.close()

    # 마지막에 한번 더 검증 (마지막 에폭에서 검증하지 않았다면)
    if not should_validate:
        logger.info("최종 검증 실시 중...")
        final_val_loss, final_val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        logger.info(
            f"최종 검증 결과: Val Loss={final_val_loss:.4f}, Val MAPE={final_val_metrics['mape']:.2f}%"
        )

    return {
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "early_stopped": patience_counter >= patience,
        "validation_count": len(
            [e for e in range(1, epoch + 1) if e % val_interval == 0 or e == epoch]
        ),
    }


def evaluate_model(model, test_loader, device, model_path, config):
    """모델 평가 - 통합 임베딩 데이터셋용 (구조 정보 포함)"""
    logger.info(f"모델 로드: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    padding_value = config.get("padding_value", -9999.0)
    criterion = MaskedMSELoss(padding_value)

    # 구조화된 결과를 위한 리스트들
    structured_predictions = []
    all_predictions = []
    all_targets = []
    total_loss = 0.0

    logger.info("테스트 시작")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # collate_fn에서 반환하는 키 이름들 사용
            features = batch["features"].to(device)  # 이미 결합된 특성
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)

            # 구조 정보 추출 (collate_fn의 키 이름들과 일치)
            timekey_hrs = batch["timekey_hrs"]
            oper_ids_lists = batch["oper_ids_lists"]
            actual_lengths = batch["actual_lengths"]
            first_oper_ids = batch["first_oper_ids"]
            last_oper_ids = batch["last_oper_ids"]

            # 모델 예측
            predictions = model(features, masks)
            loss = criterion(predictions, targets, masks)
            total_loss += loss.item()

            # CPU로 변환
            predictions_cpu = predictions.cpu()
            targets_cpu = targets.cpu()
            masks_cpu = masks.cpu()

            # 배치 내 각 샘플에 대해 구조화된 결과 생성
            batch_size = predictions_cpu.shape[0]
            for sample_idx in range(batch_size):
                timekey_hr = timekey_hrs[sample_idx]
                oper_ids = oper_ids_lists[sample_idx]
                actual_length = actual_lengths[sample_idx]
                first_oper_id = first_oper_ids[sample_idx]
                last_oper_id = last_oper_ids[sample_idx]

                sample_predictions = predictions_cpu[sample_idx]
                sample_targets = targets_cpu[sample_idx]
                sample_masks = masks_cpu[sample_idx]

                # 각 시퀀스 위치에 대해 (패딩되지 않은 위치만)
                for seq_idx in range(actual_length):
                    if seq_idx < len(sample_predictions) and not sample_masks[seq_idx]:
                        pred_val = sample_predictions[seq_idx].item()
                        target_val = sample_targets[seq_idx].item()
                        oper_id = oper_ids[seq_idx] if seq_idx < len(oper_ids) else None

                        # 개별 예측 결과 저장
                        structured_predictions.append(
                            {
                                "timekey_hr": timekey_hr,
                                "oper_id": oper_id,
                                "seq_position": seq_idx,
                                "prediction": pred_val,
                                "actual": target_val,
                                "first_oper_id": first_oper_id,
                                "last_oper_id": last_oper_id,
                                "window_length": actual_length,
                            }
                        )

                        all_predictions.append(pred_val)
                        all_targets.append(target_val)

    avg_loss = total_loss / len(test_loader)

    # 메트릭 계산
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    if len(all_predictions) == 0:
        logger.warning("예측 결과가 없습니다!")
        metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "valid_count": 0}
    else:
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # MAPE 계산 (0으로 나누기 방지)
        epsilon = 1e-8
        abs_targets = np.abs(all_targets)
        abs_errors = np.abs(all_predictions - all_targets)
        safe_targets = np.maximum(abs_targets, epsilon)
        mape = np.mean(abs_errors / safe_targets * 100)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "valid_count": len(all_predictions),
        }

    logger.info(
        f"테스트 결과: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%"
    )
    logger.info(f"구조화된 예측 결과: {len(structured_predictions):,}개")

    # 모델 정보 수집
    model_info = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "model_type": config.get("model_type", "unknown"),
    }

    # checkpoint에서 추가 정보 수집 (있는 경우)
    if "feature_dim" in checkpoint:
        model_info["feature_dim"] = checkpoint["feature_dim"]
    if "epoch" in checkpoint:
        model_info["best_epoch"] = checkpoint["epoch"]
    if "val_loss" in checkpoint:
        model_info["best_val_loss"] = checkpoint["val_loss"]

    return {
        "test_loss": avg_loss,
        "metrics": metrics,
        "predictions": all_predictions,
        "targets": all_targets,
        "structured_predictions": structured_predictions,
        "model_info": model_info,
    }
