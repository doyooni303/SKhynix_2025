#!/usr/bin/env python3
"""
시계열 공정 데이터 모델링 메인 파이프라인
전처리된 DataFrame으로부터 훈련, 검증, 테스트까지 완전 자동화

사용법:
    python main.py --dataset_config ./configs/dataset_configs.yaml --model_config ./configs/model_configs.yaml --training_config ./configs/training_configs.yaml
"""

import argparse
import pandas as pd
import torch
from datetime import datetime

from utils import load_config, save_results, logger
from dataset import TimeSeriesOperDataset, create_dataloaders
from models import create_model_from_dataloader
from train import train_model, test_model_with_structure


def run(args):
    """메인 실행 함수"""

    # 설정 파일 로드
    logger.info("=" * 60)
    logger.info("시계열 공정 데이터 모델링 파이프라인 시작")
    logger.info("=" * 60)

    dataset_config = load_config(args.dataset_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    logger.info(f"Dataset 설정 로드: {args.dataset_config}")
    logger.info(f"Model 설정 로드: {args.model_config}")
    logger.info(f"Training 설정 로드: {args.training_config}")

    # 데이터 로드 및 전처리
    data_path = dataset_config["data_path"]
    logger.info(f"Excel 데이터 로드: {data_path}")

    # Excel 파일의 모든 시트 로드 (header=1)
    excel = pd.read_excel(data_path, sheet_name=None, header=1)
    logger.info(f"Excel 시트 개수: {len(excel)}")

    # 지정된 시트들 결합
    sheet_names = dataset_config.get(
        "sheet_names", ["Data_Set1(사외)", "Data_Set2(사외)"]
    )
    logger.info(f"사용할 시트: {sheet_names}")

    total = pd.concat([excel[sheet_name] for sheet_name in sheet_names])
    logger.info(f"시트 결합 후 크기: {total.shape}")

    # Unnamed: 0 컬럼 제거
    if "Unnamed: 0" in total.columns:
        total.drop(columns="Unnamed: 0", inplace=True)
        logger.info("'Unnamed: 0' 컬럼 제거 완료")

    # y값 결측치 제거
    original_size = len(total)
    data = total[~total["y"].isna()]
    logger.info(
        f"y값 결측치 제거: {original_size:,} → {len(data):,} (-{original_size - len(data):,})"
    )

    # x값 결측치 제거 (x22-x49 컬럼)
    drop_x_features = dataset_config.get(
        "drop_x_features", [f"x{i}" for i in range(22, 49 + 1)]
    )
    if drop_x_features:
        existing_drop_features = [col for col in drop_x_features if col in data.columns]
        if existing_drop_features:
            data = data.drop(columns=existing_drop_features, axis=1)
            logger.info(
                f"x22-x49 컬럼 제거: {len(existing_drop_features)}개 컬럼 제거, 현재 크기: {data.shape}"
            )

    # 추가 불필요한 컬럼 제거
    additional_drop_columns = dataset_config.get(
        "additional_drop_columns", ["lot_cd", "oper_area"]
    )
    if additional_drop_columns:
        existing_additional_drops = [
            col for col in additional_drop_columns if col in data.columns
        ]
        if existing_additional_drops:
            data = data.drop(columns=existing_additional_drops, axis=1)
            logger.info(
                f"추가 컬럼 제거: {existing_additional_drops}, 현재 크기: {data.shape}"
            )

    # 인덱스 리셋
    data.reset_index(drop=True, inplace=True)

    # 최종 데이터 정보
    logger.info(f"최종 전처리 완료:")
    logger.info(f"  - 최종 데이터 크기: {data.shape}")
    logger.info(f"  - 컬럼 목록: {list(data.columns)}")
    logger.info(f"  - 데이터 샘플:")
    logger.info(f"{data.head(3)}")

    # Dataset 생성
    logger.info("Dataset 생성 중...")
    full_dataset = TimeSeriesOperDataset(
        df=data,  # 전처리된 데이터 사용
        categorical_columns=dataset_config["categorical_columns"],
        continuous_columns=dataset_config["continuous_columns"],
        target_column=dataset_config["target_column"],
        time_window=dataset_config.get("time_window", 24),
        embedding_dim=model_config.get("embedding_dim", 8),
        sample_id_col=dataset_config.get("sample_id_col", None),
    )

    # DataLoader 생성
    logger.info("DataLoader 생성 중...")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        full_dataset, dataset_config
    )

    # 모델 생성
    logger.info("모델 생성 중...")
    model = create_model_from_dataloader(train_dataloader, model_config)

    # GPU 설정
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"사용 디바이스: {device}")

    # 훈련
    logger.info("모델 훈련 시작...")
    training_results = train_model(
        model, train_dataloader, val_dataloader, training_config, device
    )

    # 테스트
    logger.info("모델 테스트 시작...")
    model_path = training_config.get("save_path", "best_model.pth")
    test_results = test_model_with_structure(model, test_dataloader, device, model_path)

    # 결과 저장
    output_dir = training_config.get(
        "output_dir", f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_results(test_results, output_dir)

    logger.info("=" * 60)
    logger.info("파이프라인 완료!")
    logger.info(
        f"최종 성능: RMSE={test_results['metrics']['rmse']:.4f}, MAPE={test_results['metrics']['mape']:.2f}%"
    )
    logger.info(f"결과 저장 위치: {output_dir}")
    logger.info(f"모델 저장 위치: {model_path}")
    logger.info("=" * 60)


def parse_args():
    """인자 파싱"""
    parser = argparse.ArgumentParser(description="시계열 공정 데이터 모델링 파이프라인")

    parser.add_argument(
        "--dataset_config",
        type=str,
        default="./configs/dataset.yaml",
        help="데이터셋 설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/model.yaml",
        help="모델 설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="./configs/training.yaml",
        help="훈련 설정 YAML 파일 경로",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="훈련 설정 YAML 파일 경로",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
