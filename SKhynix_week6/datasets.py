# 표준 라이브러리
from typing import Dict, List, Tuple

# 데이터 처리 라이브러리
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# PyTorch 라이브러리
import torch
from torch.utils.data import Dataset

# 로깅 라이브러리
import logging

# 로거 설정 (선택사항 - 없으면 제거하거나 다음과 같이 설정)
logger = logging.getLogger(__name__)


class CategoricalProcessor:
    """범주형 변수 임베딩을 위한 처리기"""

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.label_encoders = {}
        self.vocab_sizes = {}
        self.categorical_columns = []

    def fit(self, df: pd.DataFrame, categorical_columns: List[str]):
        """전체 데이터에 대해 범주형 인코더 학습"""
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame의 범주형 컬럼들을 숫자로 변환"""
        df_encoded = df.copy()

        for col in self.categorical_columns:
            df_encoded[col] = self.label_encoders[col].transform(
                df_encoded[col].astype(str)
            )

        return df_encoded

    def get_vocab_sizes(self) -> List[int]:
        """각 범주형 변수의 vocab_size 리스트 반환"""
        return [self.vocab_sizes[col] for col in self.categorical_columns]


class GroupedOperDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,  # 입력 데이터프레임
        categorical_columns: List[str],  # 범주형 특성 컬럼 리스트
        continuous_columns: List[str],  # 연속형 특성 컬럼 리스트
        target_column: str = "y",  # 타겟(예측 대상) 컬럼명
        group_map: Dict = None,  # 그룹 매핑 정보 (옵션)
        window_size: int = 5,  # 슬라이딩 윈도우 크기
        window_stride: int = 1,  # 윈도우 이동 간격
        padding_value: float = 0.0,  # 패딩에 사용할 값
        group_position: str = "middle",  # 그룹 결정 기준 위치 ('first', 'middle', 'last')
    ):
        # 클래스 속성으로 각 파라미터 저장
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.target_column = target_column
        self.group_map = group_map
        self.window_size = window_size
        self.window_stride = window_stride
        self.padding_value = padding_value
        self.group_position = group_position

        # group_position 파라미터 유효성 검사
        if group_position not in ["first", "middle", "last"]:
            raise ValueError(
                f"group_position must be 'first', 'middle', or 'last', got {group_position}"
            )

        # 특성 차원 계산
        self.continuous_dim = len(continuous_columns)  # 연속형 특성의 개수
        self.categorical_dim = len(categorical_columns)  # 범주형 특성의 개수

        # 데이터 전처리 (최적화)
        self._preprocess_data_optimized(df)  # 최적화된 전처리 함수 호출

        # 시퀀스 생성 (최적화)
        self._create_sequences_optimized()  # 최적화된 시퀀스 생성 함수 호출

        # 데이터셋 생성 정보 로깅
        logger.info(f"GroupedOperDataset 생성 완료:")
        logger.info(f"  - 총 시퀀스 수: {len(self.sequences)}")
        logger.info(f"  - Window 크기: {window_size}")
        logger.info(f"  - Stride: {window_stride}")
        logger.info(f"  - 전체 시퀀스 길이: {self.total_sequence_length}")
        logger.info(f"  - 최대 그룹 크기: {self.max_group_size}")
        logger.info(f"  - 그룹 기준 위치: {group_position}")

    def _preprocess_data_optimized(self, df):
        """최적화된 데이터 전처리"""
        # oper_num 추출 (벡터화) - 'oper' 접두사 제거하고 숫자만 추출
        oper_nums = df["oper_id"].str.slice(4).astype(int).values

        # NumPy 배열로 정렬 (DataFrame 정렬보다 빠름)
        timekeys = df["timekey_hr"].values  # 시간 키 배열
        # lexsort: 다중 키 정렬 (oper_num으로 먼저 정렬 후 timekey로 정렬)
        sort_indices = np.lexsort((oper_nums, timekeys))

        # 정렬된 데이터를 NumPy 배열로 직접 저장
        self.timekeys = timekeys[sort_indices]  # 정렬된 시간 키
        self.oper_nums = oper_nums[sort_indices]  # 정렬된 공정 번호
        self.oper_ids = df["oper_id"].values[sort_indices]  # 정렬된 공정 ID
        self.oper_groups = df["oper_group"].values[sort_indices]  # 정렬된 공정 그룹

        # 특성 데이터를 NumPy 배열로 (메모리 연속성)
        # 연속형 특성 배열 (float32로 변환하여 메모리 절약)
        self.continuous_array = (
            df[self.continuous_columns].values[sort_indices].astype(np.float32)
        )
        if self.categorical_columns:  # 범주형 컬럼이 있는 경우
            # 범주형 특성 배열
            self.categorical_array = (
                df[self.categorical_columns].values[sort_indices].astype(np.float32)
            )
        else:  # 범주형 컬럼이 없는 경우
            # 빈 배열 생성
            self.categorical_array = np.zeros((len(df), 0), dtype=np.float32)
        # 타겟 배열
        self.target_array = (
            df[self.target_column].values[sort_indices].astype(np.float32)
        )

        # 최대 그룹 크기 계산 (벡터화)
        unique_timekeys = np.unique(self.timekeys)  # 고유한 시간 키
        max_group_size = 0  # 최대 그룹 크기 초기화
        for tk in unique_timekeys:  # 각 시간 키에 대해
            tk_mask = self.timekeys == tk  # 현재 시간 키에 해당하는 마스크
            tk_groups = self.oper_groups[tk_mask]  # 해당 시간의 그룹들
            unique_groups, counts = np.unique(
                tk_groups, return_counts=True
            )  # 그룹별 개수 계산
            max_group_size = max(max_group_size, counts.max())  # 최대값 업데이트
        self.max_group_size = max_group_size  # 최대 그룹 크기 저장

        # 전체 시퀀스 길이 = 윈도우 크기 + 최대 그룹 크기
        self.total_sequence_length = self.window_size + self.max_group_size

        # 빠른 인덱싱을 위한 구조 (NumPy 기반)
        self.timekey_starts = {}  # 각 시간 키의 시작 인덱스
        self.timekey_ends = {}  # 각 시간 키의 끝 인덱스
        current_tk = self.timekeys[0]  # 첫 번째 시간 키
        start_idx = 0  # 시작 인덱스

        # 시간 키별 인덱스 범위 계산
        for i in range(1, len(self.timekeys)):
            if self.timekeys[i] != current_tk:  # 시간 키가 변경되면
                self.timekey_starts[current_tk] = (
                    start_idx  # 현재 키의 시작 인덱스 저장
                )
                self.timekey_ends[current_tk] = i  # 현재 키의 끝 인덱스 저장
                current_tk = self.timekeys[i]  # 새로운 시간 키로 업데이트
                start_idx = i  # 새로운 시작 인덱스
        # 마지막 시간 키 처리
        self.timekey_starts[current_tk] = start_idx
        self.timekey_ends[current_tk] = len(self.timekeys)

    def _get_group_reference_index(self, window_size: int) -> int:
        """그룹을 결정할 참조 공정의 인덱스 반환"""
        if self.group_position == "first":
            return 0  # 윈도우의 첫 번째 공정을 기준으로
        elif self.group_position == "middle":
            return window_size // 2  # 윈도우의 중간 공정을 기준으로
        elif self.group_position == "last":
            return window_size - 1  # 윈도우의 마지막 공정을 기준으로

    def _create_sequences_optimized(self):
        """최적화된 시퀀스 생성"""
        self.sequences = []  # 생성된 시퀀스를 저장할 리스트 초기화
        ref_idx = self._get_group_reference_index(self.window_size)  # 참조 인덱스 계산

        # 각 timekey에 대해 처리
        for timekey in self.timekey_starts.keys():
            start = self.timekey_starts[timekey]  # 현재 시간 키의 시작 인덱스
            end = self.timekey_ends[timekey]  # 현재 시간 키의 끝 인덱스
            tk_length = end - start  # 현재 시간 키의 데이터 길이

            # 데이터가 윈도우 크기보다 작으면 스킵
            if tk_length < self.window_size:
                continue

            # 벡터화된 window sliding
            # 생성 가능한 윈도우 개수 계산
            num_windows = (tk_length - self.window_size) // self.window_stride + 1

            # 각 윈도우에 대해 처리
            for w in range(num_windows):
                # 윈도우 시작과 끝 인덱스 계산
                window_start = start + w * self.window_stride
                window_end = window_start + self.window_size
                # 윈도우에 해당하는 인덱스 배열 생성
                window_indices = np.arange(window_start, window_end)

                # 참조 공정 정보 (직접 인덱싱)
                ref_global_idx = window_indices[ref_idx]  # 참조 인덱스의 전역 위치
                reference_oper_group = self.oper_groups[
                    ref_global_idx
                ]  # 참조 공정의 그룹
                reference_oper_id = self.oper_ids[ref_global_idx]  # 참조 공정의 ID

                # 그룹 마스크 생성 (벡터화)
                tk_indices = np.arange(start, end)  # 현재 시간의 모든 인덱스
                # 참조 그룹과 같은 그룹인지 확인하는 마스크
                group_mask = self.oper_groups[start:end] == reference_oper_group
                # 같은 그룹의 인덱스들
                group_indices_local = tk_indices[group_mask]

                # Window에 포함되지 않은 그룹 인덱스만 선택
                # setdiff1d: 첫 번째 배열에만 있고 두 번째 배열에는 없는 요소 반환
                group_indices = np.setdiff1d(group_indices_local, window_indices)

                # 시퀀스 정보 저장
                self.sequences.append(
                    {
                        "window_indices": window_indices,  # 윈도우 인덱스
                        "group_indices": group_indices,  # 그룹 인덱스
                        "reference_oper_id": reference_oper_id,  # 참조 공정 ID
                        "reference_oper_group": reference_oper_group,  # 참조 그룹
                        "timekey": timekey,  # 시간 키
                    }
                )

    def __len__(self):
        # 데이터셋의 크기 (시퀀스 개수) 반환
        return len(self.sequences)

    def __getitem__(self, idx):
        """최적화된 데이터 로딩"""
        # 주어진 인덱스의 시퀀스 가져오기
        sequence = self.sequences[idx]
        window_indices = sequence["window_indices"]  # 윈도우 인덱스
        group_indices = sequence["group_indices"]  # 그룹 인덱스

        # 그룹 크기 제한
        group_size = min(
            len(group_indices), self.max_group_size
        )  # 최대 그룹 크기로 제한
        if group_size > 0:
            group_indices = group_indices[:group_size]  # 그룹 인덱스 슬라이싱

        # 사전 할당된 텐서 (torch.zeros가 np.full보다 빠름)
        # 연속형 데이터 텐서 (패딩 값으로 초기화)
        continuous_data = torch.full(
            (self.total_sequence_length, self.continuous_dim),
            self.padding_value,
            dtype=torch.float32,
        )
        # 범주형 데이터 텐서 (0으로 초기화)
        categorical_data = torch.zeros(
            (self.total_sequence_length, self.categorical_dim), dtype=torch.float32
        )
        # 타겟 텐서 (패딩 값으로 초기화)
        targets = torch.full(
            (self.total_sequence_length,), self.padding_value, dtype=torch.float32
        )

        # Window 데이터 복사 (torch.from_numpy로 zero-copy view)
        # 윈도우 부분에 연속형 데이터 복사
        continuous_data[: self.window_size] = torch.from_numpy(
            self.continuous_array[window_indices]
        )
        if self.categorical_dim > 0:  # 범주형 특성이 있는 경우
            # 윈도우 부분에 범주형 데이터 복사
            categorical_data[: self.window_size] = torch.from_numpy(
                self.categorical_array[window_indices]
            )
        # 윈도우 부분에 타겟 데이터 복사
        targets[: self.window_size] = torch.from_numpy(
            self.target_array[window_indices]
        )

        # Group 데이터 복사
        if group_size > 0:  # 그룹 데이터가 있는 경우
            # 그룹 부분에 연속형 데이터 복사
            continuous_data[self.window_size : self.window_size + group_size] = (
                torch.from_numpy(self.continuous_array[group_indices])
            )
            if self.categorical_dim > 0:  # 범주형 특성이 있는 경우
                # 그룹 부분에 범주형 데이터 복사
                categorical_data[self.window_size : self.window_size + group_size] = (
                    torch.from_numpy(self.categorical_array[group_indices])
                )
            # 그룹 부분에 타겟 데이터 복사
            targets[self.window_size : self.window_size + group_size] = (
                torch.from_numpy(self.target_array[group_indices])
            )

        # 마스크와 position_ids (torch 텐서로 직접 생성)
        # 유효 데이터 마스크 (False로 초기화)
        masks = torch.zeros(self.total_sequence_length, dtype=torch.bool)
        # 실제 데이터가 있는 부분만 True로 설정
        masks[: self.window_size + group_size] = True

        # 위치 ID 텐서 (-1로 초기화)
        position_ids = torch.full((self.total_sequence_length,), -1, dtype=torch.long)
        # 윈도우 부분은 0으로 설정
        position_ids[: self.window_size] = 0
        if group_size > 0:  # 그룹 데이터가 있는 경우
            # 그룹 부분은 1로 설정
            position_ids[self.window_size : self.window_size + group_size] = 1

        # oper_ids 리스트 (필요한 경우만)
        # 윈도우 공정 ID 가져오기
        window_oper_ids = self.oper_ids[window_indices]
        # 전체 시퀀스 길이만큼 None으로 초기화
        oper_ids_list = [None] * self.total_sequence_length
        # 윈도우 공정 ID 채우기
        for i, oper_id in enumerate(window_oper_ids):
            oper_ids_list[i] = oper_id
        if group_size > 0:  # 그룹 데이터가 있는 경우
            # 그룹 공정 ID 가져오기
            group_oper_ids = self.oper_ids[group_indices]
            # 그룹 공정 ID 채우기
            for i, oper_id in enumerate(group_oper_ids):
                oper_ids_list[self.window_size + i] = oper_id

        # 최종 데이터 딕셔너리 반환
        return {
            "continuous_data": continuous_data,  # 연속형 특성
            "categorical_data": categorical_data,  # 범주형 특성
            "targets": targets,  # 타겟
            "masks": masks,  # 유효 데이터 마스크
            "position_ids": position_ids,  # 위치 구분자
            "sequence_lengths": self.window_size + group_size,  # 실제 시퀀스 길이
            "timekey": sequence["timekey"],  # 시간 키
            "oper_ids_list": oper_ids_list,  # 공정 ID 리스트
            "reference_oper_id": sequence["reference_oper_id"],  # 참조 공정 ID
            "reference_oper_group": sequence["reference_oper_group"],  # 참조 그룹
            "group_position": self.group_position,  # 그룹 결정 위치
        }


def custom_collate_fn(batch):
    """최적화된 배치 함수"""
    # 스택 연산 최적화 (리스트 컴프리헨션 대신 직접 스택)
    batch_size = len(batch)  # 배치 크기

    # 텐서들을 한 번에 스택
    # 각 배치 아이템의 continuous_data를 스택하여 배치 텐서 생성
    continuous_data = torch.stack([item["continuous_data"] for item in batch])
    # 각 배치 아이템의 categorical_data를 스택
    categorical_data = torch.stack([item["categorical_data"] for item in batch])
    # 각 배치 아이템의 targets를 스택
    targets = torch.stack([item["targets"] for item in batch])
    # 각 배치 아이템의 masks를 스택
    masks = torch.stack([item["masks"] for item in batch])
    # 각 배치 아이템의 position_ids를 스택
    position_ids = torch.stack([item["position_ids"] for item in batch])

    # 리스트 데이터는 그대로
    return {
        "continuous_data": continuous_data,  # 배치 연속형 데이터
        "categorical_data": categorical_data,  # 배치 범주형 데이터
        "targets": targets,  # 배치 타겟
        "masks": masks,  # 배치 마스크
        "position_ids": position_ids,  # 배치 위치 ID
        "sequence_lengths": [
            item["sequence_lengths"] for item in batch
        ],  # 시퀀스 길이 리스트
        "timekeys": [item["timekey"] for item in batch],  # 시간 키 리스트
        "oper_ids_list": [item["oper_ids_list"] for item in batch],  # 공정 ID 리스트들
        "reference_oper_ids": [
            item["reference_oper_id"] for item in batch
        ],  # 참조 공정 ID 리스트
        "reference_oper_groups": [
            item["reference_oper_group"] for item in batch
        ],  # 참조 그룹 리스트
        "group_positions": [
            item["group_position"] for item in batch
        ],  # 그룹 위치 리스트
    }


def split_data_by_days(
    df: pd.DataFrame,  # 입력 데이터프레임
    train_ratio: float = 0.8,  # 학습 데이터 비율
    val_ratio: float = 0.1,  # 검증 데이터 비율
    test_ratio: float = 0.1,  # 테스트 데이터 비율
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """날짜 기준으로 데이터를 분할"""

    # timekey_hr에서 날짜(day) 추출 (시간 정보 제거)
    # 예: 20240115 -> 202401 (년월 추출)
    df["date"] = (df["timekey_hr"].astype(int) // 100).astype(int)

    # 고유한 날짜들을 시간순으로 정렬
    unique_dates = sorted(df["date"].unique())
    total_days = len(unique_dates)  # 전체 날짜 수

    # 날짜 기준으로 분할 인덱스 계산
    train_days = int(total_days * train_ratio)  # 학습용 날짜 수
    val_days = int(total_days * val_ratio)  # 검증용 날짜 수

    # 각 분할에 해당하는 날짜 범위 설정
    train_dates = unique_dates[:train_days]  # 처음부터 train_days까지
    val_dates = unique_dates[
        train_days : train_days + val_days
    ]  # train 다음부터 val_days개
    test_dates = unique_dates[train_days + val_days :]  # 나머지 모든 날짜

    # 각 분할에 해당하는 데이터 추출
    train_df = df[df["date"].isin(train_dates)].copy()  # 학습 데이터
    val_df = df[df["date"].isin(val_dates)].copy()  # 검증 데이터
    test_df = df[df["date"].isin(test_dates)].copy()  # 테스트 데이터

    # 분할 결과 정보 로깅
    logger.info(f"날짜 기준 데이터 분할 완료:")
    logger.info(f"  - 총 날짜 수: {total_days}일")
    logger.info(f"  - Train: {len(train_dates)}일 ({len(train_df):,}행)")
    logger.info(f"  - Validation: {len(val_dates)}일 ({len(val_df):,}행)")
    logger.info(f"  - Test: {len(test_dates)}일 ({len(test_df):,}행)")
    logger.info(f"  - Train 날짜 범위: {min(train_dates)} ~ {max(train_dates)}")
    logger.info(f"  - Val 날짜 범위: {min(val_dates)} ~ {max(val_dates)}")
    logger.info(f"  - Test 날짜 범위: {min(test_dates)} ~ {max(test_dates)}")

    # 학습, 검증, 테스트 데이터프레임 반환
    return train_df, val_df, test_df
