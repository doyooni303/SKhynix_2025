import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import logging
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

# sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


# 프로젝트 모듈들 import
from utils import load_config, set_random_seeds, setup_logging
from dataset import create_dataloaders
from models import create_model_from_config_and_dataloader
from train import train_model, evaluate_model


def main():
    """메인 실행 함수"""

    parser = argparse.ArgumentParser(description="시계열 시퀀스 모델링")
    parser.add_argument("--config-dir", default="configs", help="설정 파일 디렉토리")
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train", help="실행 모드"
    )
    parser.add_argument("--model-path", default=None, help="평가용 모델 경로")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 번호")
    parser.add_argument("--exp-name", default=None, help="실험명")

    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config_dir)
    set_random_seeds(42)

    # 실험명 설정
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config.get("model_type", "lstm")
        exp_name = f"{model_type}_{timestamp}"

    # 저장 디렉토리
    save_dir = config.get("save_dir", "models")
    save_dir = os.path.join("results", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{exp_name}.pth")

    # 로깅 설정
    logger = setup_logging(
        os.path.join(save_dir, config.get("log_file", f"{exp_name}.log"))
    )

    # 디바이스 설정
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 데이터로더 생성 (통합 임베딩 방식)
    logger.info("데이터 로딩 중...")
    train_loader, val_loader, test_loader, categorical_processor, embedding_layer = (
        create_dataloaders(config)
    )

    # 모델 생성 (새로운 방식)
    logger.info("모델 생성 중...")
    model = create_model_from_config_and_dataloader(config, train_loader)

    if args.mode == "train":
        # 훈련
        logger.info("훈련 시작...")
        train_results = train_model(
            model, train_loader, val_loader, config, device, model_save_path
        )

        logger.info("훈련 완료, 테스트 시작...")
        test_results = evaluate_model(
            model, test_loader, device, model_save_path, config
        )

    else:
        # 평가
        if not args.model_path:
            raise ValueError("--model-path must be provided in eval mode")
        test_results = evaluate_model(
            model, test_loader, device, args.model_path, config
        )

    # 결과 저장
    results = {
        "exp_name": exp_name,
        "config": config,
        "test_metrics": test_results["metrics"],
        "model_info": test_results.get(
            "model_info",
            {
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model_type": config.get("model_type", "unknown"),
            },
        ),
    }

    results_path = os.path.join(save_dir, f"{exp_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"결과 저장 완료: {results_path}")

    # 예측 결과 저장
    if (
        "structured_predictions" in test_results
        and test_results["structured_predictions"]
    ):
        # 구조화된 예측 결과 저장 (timekey_hr, oper_id 포함)
        structured_df = pd.DataFrame(test_results["structured_predictions"])

        # 컬럼명 확인 후 에러 계산 (prediction vs predicted 통일)
        pred_col = (
            "prediction" if "prediction" in structured_df.columns else "predicted"
        )

        structured_df["error"] = structured_df[pred_col] - structured_df["actual"]
        structured_df["abs_error"] = structured_df["error"].abs()
        structured_df["abs_percent_error"] = (
            structured_df["abs_error"]
            / structured_df["actual"].abs().clip(lower=1e-8)
            * 100
        )

        # 구조화된 결과를 메인 예측 파일로 저장
        predictions_path = os.path.join(save_dir, f"{exp_name}_predictions.csv")
        structured_df.to_csv(predictions_path, index=False)

        logger.info(f"구조화된 예측 결과 저장:")
        logger.info(f"  - 파일 경로: {predictions_path}")
        logger.info(f"  - 저장된 예측 개수: {len(structured_df):,}개")
        logger.info(f"  - 고유한 timekey_hr: {structured_df['timekey_hr'].nunique()}개")

        # oper_id 정보 출력 (있는 경우)
        if "oper_id" in structured_df.columns:
            logger.info(f"  - 고유한 oper_id: {structured_df['oper_id'].nunique()}개")

        # 추가 구조 정보 출력 (있는 경우)
        if "first_oper_id" in structured_df.columns:
            logger.info(
                f"  - 고유한 first_oper_id: {structured_df['first_oper_id'].nunique()}개"
            )
        if "last_oper_id" in structured_df.columns:
            logger.info(
                f"  - 고유한 last_oper_id: {structured_df['last_oper_id'].nunique()}개"
            )
        if "window_length" in structured_df.columns:
            avg_window = structured_df["window_length"].mean()
            logger.info(f"  - 평균 윈도우 길이: {avg_window:.1f}")

    else:
        # 구조화된 정보가 없는 경우 기본 방식으로 저장 (호환성 유지)
        if "predictions" in test_results and "targets" in test_results:
            predictions_df = pd.DataFrame(
                {
                    "actual": test_results["targets"],
                    "predicted": test_results["predictions"],
                    "residual": test_results["targets"] - test_results["predictions"],
                    "abs_error": np.abs(
                        test_results["targets"] - test_results["predictions"]
                    ),
                    "abs_percent_error": (
                        np.abs(test_results["targets"] - test_results["predictions"])
                        / np.maximum(np.abs(test_results["targets"]), 1e-8)
                        * 100
                    ),
                }
            )

            predictions_path = os.path.join(save_dir, f"{exp_name}_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)

            logger.info(f"기본 예측 결과 저장:")
            logger.info(f"  - 파일 경로: {predictions_path}")
            logger.info(f"  - 저장된 예측 개수: {len(predictions_df):,}개")
        else:
            logger.warning("예측 결과 데이터가 없어 CSV 파일을 저장할 수 없습니다.")

    # 최종 결과 요약
    logger.info("=" * 50)
    logger.info(f"실험 완료: {exp_name}")
    logger.info(f"모델 타입: {config.get('model_type')}")
    logger.info(f"테스트 RMSE: {test_results['metrics']['rmse']:.4f}")
    logger.info(f"테스트 MAE: {test_results['metrics']['mae']:.4f}")
    logger.info(f"테스트 MAPE: {test_results['metrics']['mape']:.2f}%")

    # 모델 정보 출력
    model_info = results["model_info"]
    if "total_parameters" in model_info:
        logger.info(f"총 파라미터 수: {model_info['total_parameters']:,}")
    if "trainable_parameters" in model_info:
        logger.info(f"학습 가능한 파라미터 수: {model_info['trainable_parameters']:,}")

    logger.info(f"모델 저장 위치: {model_save_path}")
    logger.info(f"결과 저장 위치: {results_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
