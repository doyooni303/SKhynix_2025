#!/usr/bin/env python3
"""
유틸리티 함수들
설정 파일 로드, 데이터 분할, 로깅 등의 공통 기능
"""

import yaml
import logging
import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset


# 로깅 설정
def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def split_dataset_by_ratio(
    dataset, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
):
    """Dataset을 비율에 따라 순서대로 분할"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(f"Dataset 분할 완료:")
    logger.info(f"  - 전체: {total_size:,}개")
    logger.info(
        f"  - 훈련: {len(train_dataset):,}개 ({len(train_dataset)/total_size:.1%})"
    )
    logger.info(f"  - 검증: {len(val_dataset):,}개 ({len(val_dataset)/total_size:.1%})")
    logger.info(
        f"  - 테스트: {len(test_dataset):,}개 ({len(test_dataset)/total_size:.1%})"
    )

    return train_dataset, val_dataset, test_dataset


def save_results(test_results, output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # 성능 지표 저장
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(
            {
                "test_loss": test_results["test_loss"],
                "metrics": test_results["metrics"],
            },
            f,
            indent=2,
        )

    # 구조화된 예측 결과 저장
    df_predictions = pd.DataFrame(test_results["structured_predictions"])
    df_predictions["error"] = df_predictions["predicted"] - df_predictions["actual"]
    df_predictions["abs_error"] = df_predictions["error"].abs()
    df_predictions["abs_percent_error"] = (
        df_predictions["abs_error"] / df_predictions["actual"].abs().clip(lower=1e-8)
    ) * 100

    df_predictions.to_csv(f"{output_dir}/predictions.csv", index=False)

    logger.info(f"결과 저장 완료: {output_dir}")
    logger.info(f"  - metrics.json: 성능 지표")
    logger.info(f"  - predictions.csv: 예측 결과 ({len(df_predictions):,}개)")


class CategoricalProcessor:
    """범주형 변수 처리를 위한 클래스"""

    def __init__(self, embedding_dim=8):
        self.embedding_dim = embedding_dim
        self.label_encoders = {}
        self.vocab_sizes = {}
        self.categorical_columns = []

    def fit(self, df, categorical_columns):
        """범주형 컬럼들의 인코더를 학습"""
        self.categorical_columns = categorical_columns

        for col in categorical_columns:
            unique_values = df[col].astype(str).unique()
            encoder = LabelEncoder()
            encoder.fit(unique_values)

            self.label_encoders[col] = encoder
            self.vocab_sizes[col] = len(encoder.classes_)

        logger.info(f"범주형 변수별 고유값 개수:")
        for col in categorical_columns:
            logger.info(f"  {col}: {self.vocab_sizes[col]}개")

    def transform(self, df):
        """DataFrame의 범주형 컬럼들을 숫자로 변환"""
        df_encoded = df.copy()

        for col in self.categorical_columns:
            df_encoded[col] = self.label_encoders[col].transform(
                df_encoded[col].astype(str)
            )

        return df_encoded

    def get_embedding_specs(self):
        """임베딩 레이어 생성을 위한 스펙 반환"""
        return {
            col: (vocab_size, self.embedding_dim)
            for col, vocab_size in self.vocab_sizes.items()
        }
