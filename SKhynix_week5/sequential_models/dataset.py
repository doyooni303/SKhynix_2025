import os
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)
# sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def extract_oper_number(oper_id):
    """oper_id에서 숫자 부분 추출 (예: 'oper123' -> 123)"""
    match = re.search(r"\d+", str(oper_id))
    return int(match.group()) if match else 0


class CategoricalProcessor:
    """통합된 범주형 변수 처리기 - 모든 범주형 변수를 하나의 vocabulary로 통합"""

    def __init__(
        self,
        categorical_columns: List[str],
        categories: List[int],
        embedding_dim: int = 8,
    ):
        """
        Args:
            categorical_columns: 범주형 컬럼명 리스트 (예: ["oper_group", "days", "shift", "x1"])
            categories: 각 범주형 변수의 카테고리 수 (예: [277, 7, 3, 20])
            embedding_dim: 임베딩 차원
        """
        self.categorical_columns = categorical_columns
        self.categories = categories
        self.embedding_dim = embedding_dim

        # 각 변수별 인코더와 오프셋 계산
        self.label_encoders = {}
        self.category_offsets = {}  # 각 변수의 시작 인덱스
        self.category_ranges = {}  # 각 변수의 (start, end) 범위

        # 통합 vocabulary 크기 계산
        self.total_vocab_size = sum(categories)

        logger.info(f"통합된 범주형 변수 설정:")
        logger.info(
            f"  - 변수별 카테고리 수: {dict(zip(categorical_columns, categories))}"
        )
        logger.info(f"  - 총 vocabulary 크기: {self.total_vocab_size}")

    def fit(self, df: pd.DataFrame):
        """전체 데이터에 대해 통합 범주형 인코더 학습"""

        current_offset = 0

        for i, col in enumerate(self.categorical_columns):
            # 해당 변수의 고유값들 추출
            unique_values = df[col].astype(str).unique()

            # LabelEncoder 학습 (0부터 시작)
            encoder = LabelEncoder()
            encoder.fit(unique_values)

            # 실제 고유값 개수 확인
            actual_vocab_size = len(encoder.classes_)
            expected_vocab_size = self.categories[i]

            if actual_vocab_size != expected_vocab_size:
                logger.info(
                    f"  경고: {col}의 실제 카테고리 수({actual_vocab_size})가 설정값({expected_vocab_size})과 다릅니다"
                )
                # 실제값으로 업데이트
                self.categories[i] = actual_vocab_size

            self.label_encoders[col] = encoder
            self.category_offsets[col] = current_offset
            self.category_ranges[col] = (
                current_offset,
                current_offset + actual_vocab_size,
            )

            logger.info(
                f"  - {col}: 인덱스 {current_offset}~{current_offset + actual_vocab_size - 1} ({actual_vocab_size}개)"
            )

            current_offset += actual_vocab_size

        # 총 vocabulary 크기 재계산
        self.total_vocab_size = current_offset
        logger.info(f"  - 최종 통합 vocabulary 크기: {self.total_vocab_size}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame의 범주형 컬럼들을 통합 인덱스로 변환"""
        df_encoded = df.copy()

        for col in self.categorical_columns:
            # 먼저 LabelEncoder로 0부터 시작하는 인덱스로 변환
            encoded_values = self.label_encoders[col].transform(
                df_encoded[col].astype(str)
            )

            # 오프셋 추가하여 통합 vocabulary 인덱스로 변환
            df_encoded[col] = encoded_values + self.category_offsets[col]

        return df_encoded

    def get_total_vocab_size(self) -> int:
        """통합된 총 vocabulary 크기 반환"""
        return self.total_vocab_size

    def get_category_info(self) -> Dict:
        """범주형 변수 정보 딕셔너리 반환"""
        return {
            "categorical_columns": self.categorical_columns,
            "categories": self.categories,
            "category_offsets": self.category_offsets,
            "category_ranges": self.category_ranges,
            "total_vocab_size": self.total_vocab_size,
        }

    def decode_categorical_value(self, encoded_value: int) -> Tuple[str, str]:
        """통합 인덱스를 원래 (변수명, 값) 으로 디코딩"""
        for col in self.categorical_columns:
            start, end = self.category_ranges[col]
            if start <= encoded_value < end:
                # 오프셋 제거하여 원래 LabelEncoder 인덱스로 변환
                original_idx = encoded_value - self.category_offsets[col]
                # 원래 값 복원
                original_value = self.label_encoders[col].inverse_transform(
                    [original_idx]
                )[0]
                return col, original_value

        return "unknown", "unknown"


class SequenceOperDataset(Dataset):
    """시퀀스 기반 oper 데이터셋 - 통합 임베딩 + Window sliding"""

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        continuous_columns: List[str],
        categories: List[int],  # 새로운 파라미터
        target_column: str = "y",
        categorical_processor: Optional[CategoricalProcessor] = None,
        window_size: int = 30,
        stride: int = None,
        embedding_dim: int = 8,
        padding_value: float = -9999.0,
    ):
        self.df = df.copy()
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.categories = categories
        self.target_column = target_column
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.embedding_dim = embedding_dim
        self.padding_value = padding_value

        # 통합 범주형 데이터 처리기 설정
        if categorical_processor is None:
            self.categorical_processor = CategoricalProcessor(
                categorical_columns, categories, embedding_dim
            )
            self.categorical_processor.fit(df)
        else:
            self.categorical_processor = categorical_processor

        # 통합 임베딩 레이어 생성 (학습 가능)
        total_vocab_size = self.categorical_processor.get_total_vocab_size()
        self.categorical_embedding = nn.Embedding(
            total_vocab_size, embedding_dim, padding_idx=0
        )

        # 데이터 전처리 및 시퀀스 생성
        self._preprocess_data()
        self._create_windowed_sequences()

        logger.info(f"시퀀스 데이터셋 구성 완료:")
        logger.info(f"  - 총 샘플 수: {len(self.samples)}")
        logger.info(f"  - Window 크기: {window_size}")
        logger.info(f"  - Stride: {self.stride}")
        logger.info(f"  - 연속형 차원: {len(continuous_columns)}")
        logger.info(f"  - 범주형 변수 수: {len(categorical_columns)}")
        logger.info(f"  - 임베딩 차원: {embedding_dim}")
        logger.info(
            f"  - 최종 특성 차원: {len(continuous_columns) + len(categorical_columns) * embedding_dim}"
        )
        logger.info(f"  - 패딩값: {padding_value}")

    def _preprocess_data(self):
        """데이터 전처리"""
        # timekey_hr에서 날짜(day) 추출
        self.df["date"] = (self.df["timekey_hr"] // 100).astype(int)

        # 통합 범주형 데이터 인코딩
        if self.categorical_columns:
            self.df = self.categorical_processor.transform(self.df)

        # 최종 특성 차원 계산
        self.continuous_dim = len(self.continuous_columns)
        self.categorical_dim = len(self.categorical_columns) * self.embedding_dim
        self.feature_dim = self.continuous_dim + self.categorical_dim

    def _create_windowed_sequences(self):
        """timekey_hr별로 window sliding하여 시퀀스 생성"""
        self.samples = []

        # timekey_hr별로 그룹화
        grouped = self.df.groupby("timekey_hr")

        for timekey_hr, group in grouped:
            # oper_id 순서로 정렬
            group_sorted = group.iloc[
                group["oper_id"].map(extract_oper_number).argsort()
            ].reset_index(drop=True)

            if len(group_sorted) == 0:
                continue

            # Window sliding
            for start_idx in range(0, len(group_sorted), self.stride):
                end_idx = start_idx + self.window_size

                # 윈도우 데이터 추출
                window_data = group_sorted.iloc[
                    start_idx : min(end_idx, len(group_sorted))
                ]
                actual_length = len(window_data)

                if actual_length == 0:
                    continue

                # 연속형 데이터 추출 및 패딩
                if self.continuous_columns:
                    continuous_data = window_data[self.continuous_columns].values
                    if actual_length < self.window_size:
                        padding_rows = self.window_size - actual_length
                        padding_matrix = np.full(
                            (padding_rows, len(self.continuous_columns)),
                            self.padding_value,
                            dtype=np.float32,
                        )
                        continuous_data = np.vstack([continuous_data, padding_matrix])
                    continuous_data = continuous_data.astype(np.float32)
                else:
                    continuous_data = np.empty((self.window_size, 0), dtype=np.float32)

                # 범주형 데이터 추출 및 패딩 (통합 인덱스, 1부터 시작)
                if self.categorical_columns:
                    categorical_data = window_data[self.categorical_columns].values
                    if actual_length < self.window_size:
                        padding_rows = self.window_size - actual_length
                        # 패딩은 0으로 (패딩용 인덱스)
                        padding_matrix = np.zeros(
                            (padding_rows, len(self.categorical_columns)),
                            dtype=np.int64,
                        )
                        categorical_data = np.vstack([categorical_data, padding_matrix])
                    categorical_data = categorical_data.astype(np.int64)
                else:
                    categorical_data = np.empty((self.window_size, 0), dtype=np.int64)

                # 타겟 데이터 추출 및 패딩
                target_data = window_data[self.target_column].values
                if actual_length < self.window_size:
                    padding_rows = self.window_size - actual_length
                    padding_targets = np.full(
                        padding_rows, self.padding_value, dtype=np.float32
                    )
                    target_data = np.hstack([target_data, padding_targets])
                target_data = target_data.astype(np.float32)

                # 마스크 생성 (True = 패딩된 위치)
                mask = np.zeros(self.window_size, dtype=bool)
                if actual_length < self.window_size:
                    mask[actual_length:] = True

                # oper_id 정보 (전체 리스트)
                oper_ids = window_data["oper_id"].values.tolist()
                if actual_length < self.window_size:
                    oper_ids.extend([None] * (self.window_size - actual_length))

                sample_info = {
                    "timekey_hr": timekey_hr,
                    "oper_ids_list": oper_ids,
                    "continuous_data": continuous_data,
                    "categorical_data": categorical_data,
                    "target_data": target_data,
                    "mask": mask,
                    "actual_length": actual_length,
                    "first_oper_id": window_data["oper_id"].iloc[0],
                    "last_oper_id": window_data["oper_id"].iloc[-1],
                }

                self.samples.append(sample_info)

        logger.info(f"Window sliding 결과:")
        logger.info(f"  - 총 샘플 수: {len(self.samples)}")
        if self.samples:
            actual_lengths = [sample["actual_length"] for sample in self.samples]
            logger.info(f"  - 평균 실제 길이: {np.mean(actual_lengths):.1f}")
            logger.info(f"  - 최대 실제 길이: {np.max(actual_lengths)}")
            logger.info(f"  - 최소 실제 길이: {np.min(actual_lengths)}")

            timekey_hrs = [sample["timekey_hr"] for sample in self.samples]
            unique_timekey_hrs = len(set(timekey_hrs))
            logger.info(f"  - 고유한 timekey_hr: {unique_timekey_hrs}개")
            logger.info(
                f"  - timekey_hr당 평균 샘플 수: {len(self.samples)/unique_timekey_hrs:.1f}개"
            )

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):
    #     sample = self.samples[idx]

    #     # 연속형 데이터
    #     continuous_data = torch.tensor(sample['continuous_data'])  # [window_size, continuous_dim]

    #     # 범주형 데이터 → 임베딩 적용 → flatten
    #     categorical_data = torch.tensor(sample['categorical_data'])  # [window_size, num_categorical]

    #     # 임베딩 적용: [window_size, num_categorical, embed_dim]
    #     with torch.no_grad():  # 여기서는 gradient 계산 안함 (forward에서 계산)
    #         categorical_embedded = self.categorical_embedding(categorical_data)

    #     # Flatten: [window_size, num_categorical * embed_dim]
    #     window_size, num_categorical, embed_dim = categorical_embedded.shape
    #     categorical_flattened = categorical_embedded.view(window_size, num_categorical * embed_dim)

    #     # 연속형 + 범주형 결합: [window_size, total_feature_dim]
    #     if continuous_data.numel() > 0:  # 연속형 변수가 있는 경우
    #         combined_features = torch.cat([continuous_data, categorical_flattened], dim=-1)
    #     else:  # 연속형 변수가 없는 경우
    #         combined_features = categorical_flattened

    #     return {
    #         'features': combined_features,  # 최종 결합된 특성
    #         'targets': torch.tensor(sample['target_data']),
    #         'masks': torch.tensor(sample['mask']),
    #         'actual_length': sample['actual_length'],
    #         'timekey_hr': sample['timekey_hr'],
    #         'oper_ids_list': sample['oper_ids_list'],
    #         'first_oper_id': sample['first_oper_id'],
    #         'last_oper_id': sample['last_oper_id'],
    #         # 디버깅용 원본 데이터도 포함
    #         'continuous_data': continuous_data,
    #         'categorical_data': categorical_data
    #     }
    def __getitem__(self, idx):
        """디버깅 버전의 __getitem__ 메소드"""
        sample = self.samples[idx]

        continuous_data = torch.tensor(sample["continuous_data"])
        categorical_data = torch.tensor(sample["categorical_data"])

        # 임베딩 적용 전 범위 체크
        max_index = categorical_data.max()
        vocab_size = self.categorical_embedding.num_embeddings

        with torch.no_grad():
            categorical_embedded = self.categorical_embedding(categorical_data)

        # Flatten
        window_size, num_categorical, embed_dim = categorical_embedded.shape
        categorical_flattened = categorical_embedded.view(
            window_size, num_categorical * embed_dim
        )

        if continuous_data.numel() > 0:
            combined_features = torch.cat(
                [continuous_data, categorical_flattened], dim=-1
            )
        else:
            combined_features = categorical_flattened

        targets = torch.tensor(sample["target_data"])

        # 최종 반환값에 NaN/Inf가 있는지 체크
        final_nan = (
            torch.isnan(combined_features).sum()
            + torch.isnan(targets).sum()
            + torch.isinf(combined_features).sum()
            + torch.isinf(targets).sum()
        )

        if final_nan > 0:
            print(f"❌ CRITICAL: 샘플 {idx}에서 최종 NaN/Inf 발견!")
            print(f"   Features NaN: {torch.isnan(combined_features).sum()}")
            print(f"   Features Inf: {torch.isinf(combined_features).sum()}")
            print(f"   Targets NaN: {torch.isnan(targets).sum()}")
            print(f"   Targets Inf: {torch.isinf(targets).sum()}")

            # 어디서 NaN이 생겼는지 추적
            if torch.isnan(continuous_data).sum() > 0:
                print(f"   → 연속형 데이터에서 NaN 발생")
            if torch.isnan(categorical_flattened).sum() > 0:
                print(f"   → 범주형 임베딩에서 NaN 발생")

            print(f"=== 샘플 {idx} 디버깅 완료 ===\n")

        return {
            "features": combined_features,
            "targets": targets,
            "masks": torch.tensor(sample["mask"]),
            "actual_length": sample["actual_length"],
            "timekey_hr": sample["timekey_hr"],
            "oper_ids_list": sample["oper_ids_list"],
            "first_oper_id": sample["first_oper_id"],
            "last_oper_id": sample["last_oper_id"],
            "continuous_data": continuous_data,
            "categorical_data": categorical_data,
        }

    def get_embedding_layer(self) -> nn.Embedding:
        """임베딩 레이어 반환 (모델에서 사용)"""
        return self.categorical_embedding

    def get_feature_dim(self) -> int:
        """최종 특성 차원 반환"""
        return self.feature_dim


def split_data_by_days(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """날짜 기준으로 데이터를 분할"""

    # timekey_hr에서 날짜(day) 추출
    df["date"] = (df["timekey_hr"].astype(int) // 100).astype(int)

    # 고유한 날짜들을 시간순으로 정렬
    unique_dates = sorted(df["date"].unique())
    total_days = len(unique_dates)

    # 날짜 기준으로 분할 인덱스 계산
    train_days = int(total_days * train_ratio)
    val_days = int(total_days * val_ratio)

    train_dates = unique_dates[:train_days]
    val_dates = unique_dates[train_days : train_days + val_days]
    test_dates = unique_dates[train_days + val_days :]

    # 각 분할에 해당하는 데이터 추출
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    logger.info(f"날짜 기준 데이터 분할 완료:")
    logger.info(f"  - 총 날짜 수: {total_days}일")
    logger.info(f"  - Train: {len(train_dates)}일 ({len(train_df):,}행)")
    logger.info(f"  - Validation: {len(val_dates)}일 ({len(val_df):,}행)")
    logger.info(f"  - Test: {len(test_dates)}일 ({len(test_df):,}행)")
    logger.info(f"  - Train 날짜 범위: {min(train_dates)} ~ {max(train_dates)}")
    logger.info(f"  - Val 날짜 범위: {min(val_dates)} ~ {max(val_dates)}")
    logger.info(f"  - Test 날짜 범위: {min(test_dates)} ~ {max(test_dates)}")

    return train_df, val_df, test_df


def sequence_collate_fn(batch):
    """시퀀스 배치 collate 함수"""

    # 주요 텐서들 추출
    features = torch.stack([sample["features"] for sample in batch])
    targets = torch.stack([sample["targets"] for sample in batch])
    masks = torch.stack([sample["masks"] for sample in batch])

    # 메타 정보들
    actual_lengths = [sample["actual_length"] for sample in batch]
    timekey_hrs = [sample["timekey_hr"] for sample in batch]
    oper_ids_lists = [sample["oper_ids_list"] for sample in batch]
    first_oper_ids = [sample["first_oper_id"] for sample in batch]
    last_oper_ids = [sample["last_oper_id"] for sample in batch]

    # 디버깅용 원본 데이터
    continuous_data = torch.stack([sample["continuous_data"] for sample in batch])
    categorical_data = torch.stack([sample["categorical_data"] for sample in batch])

    return {
        "features": features,  # 최종 결합된 특성 [batch_size, window_size, total_feature_dim]
        "targets": targets,
        "masks": masks,
        "actual_lengths": actual_lengths,
        "timekey_hrs": timekey_hrs,
        "oper_ids_lists": oper_ids_lists,
        "first_oper_ids": first_oper_ids,
        "last_oper_ids": last_oper_ids,
        # 디버깅용
        "continuous_data": continuous_data,
        "categorical_data": categorical_data,
    }


def create_dataloaders(
    dataset_config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, CategoricalProcessor, nn.Embedding]:
    """데이터로더 생성 - 날짜 기준 분할 + 통합 임베딩 + Window sliding"""

    # 데이터 로드
    data_path = dataset_config["file_path"]
    excel = pd.read_excel(data_path, sheet_name=None, header=1)
    sheet_names = dataset_config["sheet_names"]

    total_df = pd.concat([excel[sheet_name] for sheet_name in sheet_names])

    # 기본 전처리
    if "Unnamed: 0" in total_df.columns:
        total_df.drop(columns="Unnamed: 0", inplace=True)

    # y값 결측치 제거
    df = total_df[~total_df[dataset_config["target_column"]].isna()].copy()
    logger.info(f"y값 제거 후: {len(df)}행")

    # ✅ Inf 값 확인 및 처리 추가
    logger.info("\nInf 값 확인:")
    continuous_cols = dataset_config["continuous_columns"]
    inf_found = False

    for col in continuous_cols:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                print(f"  ❌ {col}: {inf_count}개 Inf 값")
                inf_found = True

                # Inf 값을 NaN으로 변환 후 적절한 값으로 대체
                df.loc[np.isinf(df[col]), col] = np.nan
                df[col] = df[col].fillna(1e5)  # 중간값으로 대체
                logger.info(f"     → 아주 큰 값({1e+5})으로 대체")

    if not inf_found:
        print("  ✅ 연속형 변수에 Inf 없음")

    # 불필요한 컬럼 제거
    drop_columns = dataset_config.get("additional_drop_columns", [])
    if drop_columns:
        existing_drops = [col for col in drop_columns if col in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops)

    df.reset_index(drop=True, inplace=True)

    logger.info(f"원본 데이터 로드 완료:")
    logger.info(f"  - 총 행 수: {len(df):,}개")
    logger.info(f"  - 고유 timekey_hr: {df['timekey_hr'].nunique()}개")

    # 통합 범주형 처리기 생성 및 학습
    categorical_processor = CategoricalProcessor(
        categorical_columns=dataset_config["categorical_columns"],
        categories=dataset_config.get("categories", []),  # 새로운 파라미터
        embedding_dim=dataset_config.get("embedding_dim", 8),
    )
    categorical_processor.fit(df)

    # 날짜 기준으로 데이터 분할
    train_df, val_df, test_df = split_data_by_days(
        df,
        train_ratio=dataset_config.get("train_ratio", 0.8),
        val_ratio=dataset_config.get("val_ratio", 0.1),
        test_ratio=dataset_config.get("test_ratio", 0.1),
    )

    # Window sliding 파라미터
    window_size = dataset_config.get("window_size", 30)
    stride = dataset_config.get("stride", None)
    padding_value = dataset_config.get("padding_value", -9999.0)

    logger.info(f"\nWindow sliding 설정:")
    logger.info(f"  - Window 크기: {window_size}")
    logger.info(f"  - Stride: {stride if stride is not None else window_size} (비겹침)")
    logger.info(f"  - Padding 값: {padding_value}")

    # 데이터셋 생성
    train_dataset = SequenceOperDataset(
        df=train_df,
        categorical_columns=dataset_config["categorical_columns"],
        continuous_columns=dataset_config["continuous_columns"],
        categories=dataset_config.get("categories", []),
        target_column=dataset_config["target_column"],
        categorical_processor=categorical_processor,
        window_size=window_size,
        stride=stride,
        embedding_dim=dataset_config.get("embedding_dim", 8),
        padding_value=padding_value,
    )

    val_dataset = SequenceOperDataset(
        df=val_df,
        categorical_columns=dataset_config["categorical_columns"],
        continuous_columns=dataset_config["continuous_columns"],
        categories=dataset_config.get("categories", []),
        target_column=dataset_config["target_column"],
        categorical_processor=categorical_processor,
        window_size=window_size,
        stride=stride,
        embedding_dim=dataset_config.get("embedding_dim", 8),
        padding_value=padding_value,
    )

    test_dataset = SequenceOperDataset(
        df=test_df,
        categorical_columns=dataset_config["categorical_columns"],
        continuous_columns=dataset_config["continuous_columns"],
        categories=dataset_config.get("categories", []),
        target_column=dataset_config["target_column"],
        categorical_processor=categorical_processor,
        window_size=window_size,
        stride=stride,
        embedding_dim=dataset_config.get("embedding_dim", 8),
        padding_value=padding_value,
    )

    # 임베딩 레이어 추출 (모델에서 사용할 용도)
    embedding_layer = train_dataset.get_embedding_layer()

    # 데이터로더 생성
    batch_size = dataset_config.get("batch_size", 32)
    num_workers = dataset_config.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"\n데이터로더 생성 완료:")
    logger.info(f"  - 배치 크기: {batch_size}")
    logger.info(f"  - Train 배치 수: {len(train_loader)}")
    logger.info(f"  - Val 배치 수: {len(val_loader)}")
    logger.info(f"  - Test 배치 수: {len(test_loader)}")

    # 특성 차원 정보 출력
    feature_dim = train_dataset.get_feature_dim()
    logger.info(f"  - 최종 특성 차원: {feature_dim}")

    return train_loader, val_loader, test_loader, categorical_processor, embedding_layer
