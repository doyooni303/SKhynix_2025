#!/usr/bin/env python3
"""
Dataset 및 DataLoader 관련 클래스들
TimeSeriesOperDataset, Collate 함수, DataLoader 생성 등
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import CategoricalProcessor, split_dataset_by_ratio, logger


class TimeSeriesOperDataset(Dataset):
    """시간대별 oper 데이터를 위한 Dataset (구조 정보 보존)"""

    def __init__(
        self,
        df,
        categorical_columns,
        continuous_columns,
        target_column="y",
        categorical_processor=None,
        time_window=24,
        embedding_dim=8,
        sample_id_col=None,
        global_max_oper=None,  # 전체 데이터셋의 max_oper 값을 받아서 사용
    ):
        self.df = df.copy()
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.target_column = target_column
        self.time_window = time_window
        self.embedding_dim = embedding_dim
        self.sample_id_col = sample_id_col

        # 범주형 데이터 처리기 설정
        if categorical_processor is None:
            self.categorical_processor = CategoricalProcessor(embedding_dim)
            self.categorical_processor.fit(df, categorical_columns)
        else:
            self.categorical_processor = categorical_processor

        # 데이터 전처리 - 안전한 인덱스 처리
        self.df_encoded = self.df.copy()
        if categorical_columns:
            categorical_encoded = self.categorical_processor.transform(
                self.df[categorical_columns]
            )
            # 안전성 검증: 인코딩된 값이 유효 범위 내에 있는지 확인
            for col in categorical_columns:
                max_value = categorical_encoded[col].max()
                vocab_size = len(
                    self.categorical_processor.label_encoders[col].classes_
                )
                if max_value >= vocab_size:
                    logger.warning(
                        f"열 '{col}'에서 범위 초과 값 발견: max={max_value}, vocab_size={vocab_size}"
                    )
                    # 범위를 벗어난 값을 0(unknown)으로 클램핑
                    categorical_encoded[col] = categorical_encoded[col].clip(
                        0, vocab_size - 1
                    )

            self.df_encoded[categorical_columns] = categorical_encoded

        # 전체 데이터셋의 max_oper를 계산하거나 받아서 사용
        if global_max_oper is None:
            self.global_max_oper = self._calculate_global_max_oper()
        else:
            self.global_max_oper = global_max_oper

        # 시간 기반 샘플 분할
        self._prepare_time_based_samples()

        logger.info(f"Dataset 구성 완료:")
        logger.info(f"  - 총 샘플 수: {len(self.samples)}")
        logger.info(f"  - 시간 윈도우 크기: {time_window}")
        logger.info(f"  - 전체 데이터셋 max_oper_per_hour: {self.global_max_oper}")
        logger.info(
            f"  - 입력 변수: 범주형 {len(categorical_columns)}개 + 연속형 {len(continuous_columns)}개"
        )
        logger.info(f"  - 출력 변수: {target_column}")
        logger.info(f"  - 구조 정보 보존: timekey_hr + oper_id")

    def _calculate_global_max_oper(self):
        """전체 데이터셋에서 시간당 최대 oper 수를 계산"""
        if self.sample_id_col is not None:
            max_oper_counts = []
            sample_ids = self.df_encoded[self.sample_id_col].unique()
            for sample_id in sample_ids:
                sample_df = self.df_encoded[
                    self.df_encoded[self.sample_id_col] == sample_id
                ]
                if len(sample_df) > 0:
                    oper_counts_per_time = sample_df.groupby("timekey_hr").size()
                    max_oper_counts.append(oper_counts_per_time.max())
        else:
            if len(self.df_encoded) > 0:
                oper_counts_per_time = self.df_encoded.groupby("timekey_hr").size()
                max_oper_counts = [oper_counts_per_time.max()]
            else:
                max_oper_counts = [1]

        return max(max_oper_counts) if max_oper_counts else 1

    def _prepare_time_based_samples(self):
        """연속된 시간을 time_window 크기로 분할하여 샘플 생성"""
        self.samples = []

        if self.sample_id_col is not None:
            sample_ids = self.df_encoded[self.sample_id_col].unique()
            for sample_id in sample_ids:
                sample_df = self.df_encoded[
                    self.df_encoded[self.sample_id_col] == sample_id
                ]
                time_samples = self._split_by_time_window(sample_df)
                self.samples.extend(time_samples)
        else:
            time_samples = self._split_by_time_window(self.df_encoded)
            self.samples.extend(time_samples)

    def _split_by_time_window(self, df):
        """DataFrame을 time_window 크기로 분할하여 샘플 생성"""
        if len(df) == 0:
            return []

        # 시간순 정렬
        df = df.sort_values("timekey_hr")
        unique_times = sorted(df["timekey_hr"].unique())
        samples = []

        # 슬라이딩 윈도우 방식으로 샘플 생성
        for start_idx in range(len(unique_times) - self.time_window + 1):
            window_times = unique_times[start_idx : start_idx + self.time_window]

            # 시간 연속성 체크 (선택적)
            if self._check_time_continuity(window_times):
                window_df = df[df["timekey_hr"].isin(window_times)]

                if len(window_df) > 0:
                    sample_data = self._create_windowed_sample(window_df, window_times)
                    if sample_data is not None:
                        samples.append(sample_data)

        return samples

    def _check_time_continuity(self, time_list):
        """시간 연속성 체크 (선택적 - 필요에 따라 활성화)"""
        # 모든 윈도우를 허용하려면 True 반환
        return True

        # 엄격한 연속성 체크가 필요한 경우:
        # for i in range(1, len(time_list)):
        #     if time_list[i] - time_list[i-1] > 1:  # 1시간 초과 간격
        #         return False
        # return True

    def _create_windowed_sample(self, df, window_times):
        """
        시간 윈도우 기반 단일 샘플 데이터 구성 (기존 구조 유지)
        기존 코드 구조와 완전히 호환되도록 구성
        """
        try:
            grouped = df.groupby("timekey_hr")
            actual_time_steps = len(window_times)

            continuous_data = []
            categorical_data = []
            target_data = []
            hour_oper_counts = []
            timekey_hr_info = []
            oper_id_info = []

            # 각 시간 단위별로 데이터 처리
            for time_key in window_times:
                if time_key in grouped.groups:
                    time_data = grouped.get_group(time_key)
                    time_data = time_data.sort_values("oper_id")

                    # 연속형 데이터
                    time_continuous = (
                        time_data[self.continuous_columns].values
                        if self.continuous_columns
                        else np.empty((len(time_data), 0))
                    )

                    # 범주형 데이터 - 안전성 검증 추가
                    if self.categorical_columns:
                        time_categorical = time_data[self.categorical_columns].values
                        # 범주형 데이터의 값이 유효 범위 내에 있는지 확인
                        for col_idx, col_name in enumerate(self.categorical_columns):
                            vocab_size = len(
                                self.categorical_processor.label_encoders[
                                    col_name
                                ].classes_
                            )
                            time_categorical[:, col_idx] = np.clip(
                                time_categorical[:, col_idx], 0, vocab_size - 1
                            )
                    else:
                        time_categorical = np.empty((len(time_data), 0))

                    # 타겟 데이터
                    time_target = time_data[self.target_column].values

                    # oper_id 정보
                    time_oper_ids = time_data["oper_id"].values

                    # 해당 시간의 데이터 개수
                    hour_oper_counts.append(len(time_data))
                    timekey_hr_info.append(time_key)
                    oper_id_info.append(time_oper_ids)
                else:
                    # 해당 시간에 데이터가 없는 경우 빈 배열 생성
                    time_continuous = (
                        np.empty((0, len(self.continuous_columns)))
                        if self.continuous_columns
                        else np.empty((0, 0))
                    )
                    time_categorical = (
                        np.empty((0, len(self.categorical_columns)), dtype=np.int64)
                        if self.categorical_columns
                        else np.empty((0, 0))
                    )
                    time_target = np.empty(0)

                    hour_oper_counts.append(0)
                    timekey_hr_info.append(time_key)
                    oper_id_info.append(np.array([], dtype=int))

                continuous_data.append(time_continuous)
                categorical_data.append(time_categorical)
                target_data.append(time_target)

            # time_window 크기만큼 패딩 처리
            while len(hour_oper_counts) < self.time_window:
                hour_oper_counts.append(0)
                continuous_data.append(
                    np.empty((0, len(self.continuous_columns)))
                    if self.continuous_columns
                    else np.empty((0, 0))
                )
                categorical_data.append(
                    np.empty((0, len(self.categorical_columns)), dtype=np.int64)
                    if self.categorical_columns
                    else np.empty((0, 0))
                )
                target_data.append(np.empty(0))
                timekey_hr_info.append(None)
                oper_id_info.append(np.array([], dtype=int))

            # 기존 구조와 완전히 동일한 형태로 반환
            return {
                "continuous_data": continuous_data,
                "categorical_data": categorical_data,
                "target_data": target_data,
                "hour_oper_counts": hour_oper_counts,
                "max_oper_per_hour": self.global_max_oper,  # 전체 데이터셋의 값 사용
                "actual_time_steps": actual_time_steps,
                "window_times": list(window_times),
                "timekey_hr_info": timekey_hr_info,
                "oper_id_info": oper_id_info,
            }

        except Exception as e:
            logger.error(f"샘플 생성 중 오류 발생: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 기존 코드에서 기대하는 정확한 키 구조로 반환
        sample = self.samples[idx]
        return {
            "continuous_data": sample["continuous_data"],  # 리스트[numpy배열]
            "categorical_data": sample["categorical_data"],  # 리스트[numpy배열]
            "target_data": sample["target_data"],  # 리스트[numpy배열]
            "hour_oper_counts": sample["hour_oper_counts"],  # 리스트[정수]
            "max_oper_per_hour": sample["max_oper_per_hour"],  # 정수
            "actual_time_steps": sample["actual_time_steps"],  # 정수
            "window_times": sample["window_times"],  # 리스트[정수]
            "timekey_hr_info": sample["timekey_hr_info"],  # 리스트
            "oper_id_info": sample["oper_id_info"],  # 리스트[numpy배열]
        }


# def pad_batch_samples(
#     batch_samples,
#     continuous_padding_value=0.0,
#     categorical_padding_value=0,
#     target_padding_value=0.0,
#     global_max_oper=None,  # 전체 데이터셋의 max_oper 값
# ):
#     """배치 내 샘플들을 동일한 크기로 패딩 (구조 정보 포함)"""
#     batch_size = len(batch_samples)
#     time_window = len(batch_samples[0]["continuous_data"])

#     # global_max_oper가 주어지면 그것을 사용, 아니면 배치 내 최대값 사용
#     if global_max_oper is not None:
#         batch_max_oper = global_max_oper
#     else:
#         batch_max_oper = max(sample["max_oper_per_hour"] for sample in batch_samples)

#     first_sample = batch_samples[0]

#     # 차원 정보 추출
#     continuous_dim = 0
#     for hour_data in first_sample["continuous_data"]:
#         if len(hour_data) > 0:
#             continuous_dim = len(hour_data[0])
#             break

#     categorical_dim = 0
#     for hour_data in first_sample["categorical_data"]:
#         if len(hour_data) > 0:
#             categorical_dim = len(hour_data[0])
#             break

#     # 배치 데이터 초기화
#     batch_continuous = (
#         np.full(
#             (batch_size, time_window, batch_max_oper, continuous_dim),
#             continuous_padding_value,
#             dtype=np.float32,
#         )
#         if continuous_dim > 0
#         else np.empty((batch_size, time_window, batch_max_oper, 0))
#     )

#     batch_categorical = (
#         np.full(
#             (batch_size, time_window, batch_max_oper, categorical_dim),
#             categorical_padding_value,
#             dtype=np.int64,
#         )
#         if categorical_dim > 0
#         else np.empty((batch_size, time_window, batch_max_oper, 0))
#     )

#     batch_targets = np.full(
#         (batch_size, time_window, batch_max_oper),
#         target_padding_value,
#         dtype=np.float32,
#     )

#     batch_masks = np.ones((batch_size, time_window, batch_max_oper), dtype=bool)
#     batch_time_masks = np.zeros((batch_size, time_window), dtype=bool)

#     batch_hour_counts = []
#     batch_actual_time_steps = []
#     batch_window_times = []
#     batch_timekey_hr_info = []
#     batch_oper_id_info = []

#     for batch_idx, sample in enumerate(batch_samples):
#         continuous_data = sample["continuous_data"]
#         categorical_data = sample["categorical_data"]
#         target_data = sample["target_data"]
#         hour_counts = sample["hour_oper_counts"]
#         actual_time_steps = sample["actual_time_steps"]
#         window_times = sample["window_times"]
#         timekey_hr_info = sample["timekey_hr_info"]
#         oper_id_info = sample["oper_id_info"]

#         batch_hour_counts.append(hour_counts)
#         batch_actual_time_steps.append(actual_time_steps)
#         batch_window_times.append(window_times)
#         batch_timekey_hr_info.append(timekey_hr_info)
#         batch_oper_id_info.append(oper_id_info)

#         batch_time_masks[batch_idx, :actual_time_steps] = False
#         batch_time_masks[batch_idx, actual_time_steps:] = True

#         for time_idx in range(time_window):
#             time_oper_count = hour_counts[time_idx]

#             if time_oper_count > 0:
#                 actual_count = min(time_oper_count, batch_max_oper)

#                 if continuous_dim > 0:
#                     batch_continuous[batch_idx, time_idx, :actual_count] = (
#                         continuous_data[time_idx][:actual_count]
#                     )

#                 if categorical_dim > 0:
#                     # 범주형 데이터 안전성 검증
#                     cat_data = categorical_data[time_idx][:actual_count]
#                     # 패딩값은 0으로 설정하고, 유효하지 않은 값들을 클램핑
#                     cat_data = np.clip(
#                         cat_data, 0, 10000
#                     )  # 큰 값으로 클램핑 후 padding_value로 처리
#                     batch_categorical[batch_idx, time_idx, :actual_count] = cat_data

#                 batch_targets[batch_idx, time_idx, :actual_count] = target_data[
#                     time_idx
#                 ][:actual_count]

#                 batch_masks[batch_idx, time_idx, :actual_count] = False

#     result = {
#         "continuous_data": torch.tensor(batch_continuous),
#         "categorical_data": torch.tensor(batch_categorical),
#         "targets": torch.tensor(batch_targets),
#         "masks": torch.tensor(batch_masks),
#         "time_masks": torch.tensor(batch_time_masks),
#         "hour_counts": batch_hour_counts,
#         "actual_time_steps": batch_actual_time_steps,
#         "window_times": batch_window_times,
#         "max_oper_per_hour": batch_max_oper,
#         "timekey_hr_info": batch_timekey_hr_info,
#         "oper_id_info": batch_oper_id_info,
#     }
#
# return result


def pad_batch_samples(
    batch_samples,
    continuous_padding_value=0.0,
    categorical_padding_value=0,  # 0으로 변경 (임베딩의 padding_idx와 일치)
    target_padding_value=0.0,
    global_max_oper=None,  # 전체 데이터셋의 max_oper 값
):
    """배치 내 샘플들을 동일한 크기로 패딩 (구조 정보 포함)"""
    batch_size = len(batch_samples)
    time_window = len(batch_samples[0]["continuous_data"])

    # global_max_oper가 주어지면 그것을 사용, 아니면 배치 내 최대값 사용
    if global_max_oper is not None:
        batch_max_oper = global_max_oper
    else:
        batch_max_oper = max(sample["max_oper_per_hour"] for sample in batch_samples)

    first_sample = batch_samples[0]

    # 차원 정보 추출
    continuous_dim = 0
    for hour_data in first_sample["continuous_data"]:
        if len(hour_data) > 0:
            continuous_dim = len(hour_data[0])
            break

    categorical_dim = 0
    for hour_data in first_sample["categorical_data"]:
        if len(hour_data) > 0:
            categorical_dim = len(hour_data[0])
            break

    # 배치 데이터 초기화
    batch_continuous = (
        np.full(
            (batch_size, time_window, batch_max_oper, continuous_dim),
            continuous_padding_value,
            dtype=np.float32,
        )
        if continuous_dim > 0
        else np.empty((batch_size, time_window, batch_max_oper, 0))
    )

    batch_categorical = (
        np.full(
            (batch_size, time_window, batch_max_oper, categorical_dim),
            categorical_padding_value,  # 이제 0으로 설정됨
            dtype=np.int64,
        )
        if categorical_dim > 0
        else np.empty((batch_size, time_window, batch_max_oper, 0))
    )

    batch_targets = np.full(
        (batch_size, time_window, batch_max_oper),
        target_padding_value,
        dtype=np.float32,
    )

    batch_masks = np.ones((batch_size, time_window, batch_max_oper), dtype=bool)
    batch_time_masks = np.zeros((batch_size, time_window), dtype=bool)

    batch_hour_counts = []
    batch_actual_time_steps = []
    batch_window_times = []
    batch_timekey_hr_info = []
    batch_oper_id_info = []

    for batch_idx, sample in enumerate(batch_samples):
        continuous_data = sample["continuous_data"]
        categorical_data = sample["categorical_data"]
        target_data = sample["target_data"]
        hour_counts = sample["hour_oper_counts"]
        actual_time_steps = sample["actual_time_steps"]
        window_times = sample["window_times"]
        timekey_hr_info = sample["timekey_hr_info"]
        oper_id_info = sample["oper_id_info"]

        batch_hour_counts.append(hour_counts)
        batch_actual_time_steps.append(actual_time_steps)
        batch_window_times.append(window_times)
        batch_timekey_hr_info.append(timekey_hr_info)
        batch_oper_id_info.append(oper_id_info)

        batch_time_masks[batch_idx, :actual_time_steps] = False
        batch_time_masks[batch_idx, actual_time_steps:] = True

        for time_idx in range(time_window):
            time_oper_count = hour_counts[time_idx]

            if time_oper_count > 0:
                actual_count = min(time_oper_count, batch_max_oper)

                if continuous_dim > 0:
                    batch_continuous[batch_idx, time_idx, :actual_count] = (
                        continuous_data[time_idx][:actual_count]
                    )

                if categorical_dim > 0:
                    # 범주형 데이터 안전성 검증 및 클램핑
                    cat_data = categorical_data[time_idx][:actual_count]
                    # 음수값을 0으로, 너무 큰 값을 제한
                    cat_data = np.clip(cat_data, 0, 10000)  # 임시 상한값
                    batch_categorical[batch_idx, time_idx, :actual_count] = cat_data

                batch_targets[batch_idx, time_idx, :actual_count] = target_data[
                    time_idx
                ][:actual_count]

                batch_masks[batch_idx, time_idx, :actual_count] = False

    result = {
        "continuous_data": torch.tensor(batch_continuous),
        "categorical_data": torch.tensor(batch_categorical),
        "targets": torch.tensor(batch_targets),
        "masks": torch.tensor(batch_masks),
        "time_masks": torch.tensor(batch_time_masks),
        "hour_counts": batch_hour_counts,
        "actual_time_steps": batch_actual_time_steps,
        "window_times": batch_window_times,
        "max_oper_per_hour": batch_max_oper,
        "timekey_hr_info": batch_timekey_hr_info,
        "oper_id_info": batch_oper_id_info,
    }

    return result


class TimeSeriesOperCollate:
    """배치 패딩을 위한 Collate 클래스"""

    def __init__(
        self,
        continuous_padding_value=0.0,
        categorical_padding_value=0,
        target_padding_value=0.0,
        global_max_oper=None,  # 전체 데이터셋의 max_oper 값
    ):
        self.continuous_padding_value = continuous_padding_value
        self.categorical_padding_value = categorical_padding_value
        self.target_padding_value = target_padding_value
        self.global_max_oper = global_max_oper

    def __call__(self, batch):
        return pad_batch_samples(
            batch,
            continuous_padding_value=self.continuous_padding_value,
            categorical_padding_value=self.categorical_padding_value,
            target_padding_value=self.target_padding_value,
            global_max_oper=self.global_max_oper,
        )


# def create_dataloaders(full_dataset, dataset_config: dict):
#     """전체 Dataset으로부터 훈련/검증/테스트 DataLoader 생성"""

#     batch_size = dataset_config.get("batch_size", 32)
#     train_ratio = dataset_config.get("train_ratio", 0.7)
#     val_ratio = dataset_config.get("val_ratio", 0.15)
#     test_ratio = dataset_config.get("test_ratio", 0.15)
#     num_workers = dataset_config.get("num_workers", 4)

#     # 전체 데이터셋의 global_max_oper 값을 미리 저장
#     global_max_oper = full_dataset.global_max_oper
#     categorical_processor = full_dataset.categorical_processor

#     # 데이터셋 분할
#     train_dataset, val_dataset, test_dataset = split_dataset_by_ratio(
#         full_dataset, train_ratio, val_ratio, test_ratio
#     )

#     # 분할된 각 데이터셋에 global_max_oper와 categorical_processor를 설정
#     # 이를 위해서는 분할된 데이터셋도 TimeSeriesOperDataset으로 다시 생성해야 함
#     train_df = (
#         train_dataset.dataset.df.iloc[train_dataset.indices]
#         if hasattr(train_dataset, "indices")
#         else train_dataset.df
#     )
#     val_df = (
#         val_dataset.dataset.df.iloc[val_dataset.indices]
#         if hasattr(val_dataset, "indices")
#         else val_dataset.df
#     )
#     test_df = (
#         test_dataset.dataset.df.iloc[test_dataset.indices]
#         if hasattr(test_dataset, "indices")
#         else test_dataset.df
#     )

#     # 새로운 데이터셋 생성 (global_max_oper와 categorical_processor 공유)
#     train_dataset_new = TimeSeriesOperDataset(
#         df=train_df,
#         categorical_columns=full_dataset.categorical_columns,
#         continuous_columns=full_dataset.continuous_columns,
#         target_column=full_dataset.target_column,
#         categorical_processor=categorical_processor,
#         time_window=full_dataset.time_window,
#         embedding_dim=full_dataset.embedding_dim,
#         sample_id_col=full_dataset.sample_id_col,
#         global_max_oper=global_max_oper,
#     )

#     val_dataset_new = TimeSeriesOperDataset(
#         df=val_df,
#         categorical_columns=full_dataset.categorical_columns,
#         continuous_columns=full_dataset.continuous_columns,
#         target_column=full_dataset.target_column,
#         categorical_processor=categorical_processor,
#         time_window=full_dataset.time_window,
#         embedding_dim=full_dataset.embedding_dim,
#         sample_id_col=full_dataset.sample_id_col,
#         global_max_oper=global_max_oper,
#     )

#     test_dataset_new = TimeSeriesOperDataset(
#         df=test_df,
#         categorical_columns=full_dataset.categorical_columns,
#         continuous_columns=full_dataset.continuous_columns,
#         target_column=full_dataset.target_column,
#         categorical_processor=categorical_processor,
#         time_window=full_dataset.time_window,
#         embedding_dim=full_dataset.embedding_dim,
#         sample_id_col=full_dataset.sample_id_col,
#         global_max_oper=global_max_oper,
#     )

#     collate_fn = TimeSeriesOperCollate(
#         continuous_padding_value=dataset_config.get("continuous_padding_value", 1e4),
#         categorical_padding_value=dataset_config.get("categorical_padding_value", 1e4),
#         target_padding_value=dataset_config.get("target_padding_value", 0.0),
#         global_max_oper=global_max_oper,
#     )

#     train_dataloader = DataLoader(
#         train_dataset_new,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     val_dataloader = DataLoader(
#         val_dataset_new,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     test_dataloader = DataLoader(
#         test_dataset_new,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     logger.info(f"DataLoader 생성 완료:")
#     logger.info(f"  - 배치 크기: {batch_size}")
#     logger.info(f"  - 전체 데이터셋 global_max_oper: {global_max_oper}")
#     logger.info(f"  - 훈련 배치 수: {len(train_dataloader)}")
#     logger.info(f"  - 검증 배치 수: {len(val_dataloader)}")
#     logger.info(f"  - 테스트 배치 수: {len(test_dataloader)}")

#     return train_dataloader, val_dataloader, test_dataloader


def create_dataloaders(full_dataset, dataset_config: dict):
    """전체 Dataset으로부터 훈련/검증/테스트 DataLoader 생성"""

    batch_size = dataset_config.get("batch_size", 32)
    train_ratio = dataset_config.get("train_ratio", 0.7)
    val_ratio = dataset_config.get("val_ratio", 0.15)
    test_ratio = dataset_config.get("test_ratio", 0.15)
    num_workers = dataset_config.get("num_workers", 4)

    # 전체 데이터셋의 global_max_oper 값을 미리 저장
    global_max_oper = full_dataset.global_max_oper
    categorical_processor = full_dataset.categorical_processor

    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = split_dataset_by_ratio(
        full_dataset, train_ratio, val_ratio, test_ratio
    )

    # 분할된 각 데이터셋에 global_max_oper와 categorical_processor를 설정
    # 이를 위해서는 분할된 데이터셋도 TimeSeriesOperDataset으로 다시 생성해야 함
    train_df = (
        train_dataset.dataset.df.iloc[train_dataset.indices]
        if hasattr(train_dataset, "indices")
        else train_dataset.df
    )
    val_df = (
        val_dataset.dataset.df.iloc[val_dataset.indices]
        if hasattr(val_dataset, "indices")
        else val_dataset.df
    )
    test_df = (
        test_dataset.dataset.df.iloc[test_dataset.indices]
        if hasattr(test_dataset, "indices")
        else test_dataset.df
    )

    # 새로운 데이터셋 생성 (global_max_oper와 categorical_processor 공유)
    train_dataset_new = TimeSeriesOperDataset(
        df=train_df,
        categorical_columns=full_dataset.categorical_columns,
        continuous_columns=full_dataset.continuous_columns,
        target_column=full_dataset.target_column,
        categorical_processor=categorical_processor,
        time_window=full_dataset.time_window,
        embedding_dim=full_dataset.embedding_dim,
        sample_id_col=full_dataset.sample_id_col,
        global_max_oper=global_max_oper,
    )

    val_dataset_new = TimeSeriesOperDataset(
        df=val_df,
        categorical_columns=full_dataset.categorical_columns,
        continuous_columns=full_dataset.continuous_columns,
        target_column=full_dataset.target_column,
        categorical_processor=categorical_processor,
        time_window=full_dataset.time_window,
        embedding_dim=full_dataset.embedding_dim,
        sample_id_col=full_dataset.sample_id_col,
        global_max_oper=global_max_oper,
    )

    test_dataset_new = TimeSeriesOperDataset(
        df=test_df,
        categorical_columns=full_dataset.categorical_columns,
        continuous_columns=full_dataset.continuous_columns,
        target_column=full_dataset.target_column,
        categorical_processor=categorical_processor,
        time_window=full_dataset.time_window,
        embedding_dim=full_dataset.embedding_dim,
        sample_id_col=full_dataset.sample_id_col,
        global_max_oper=global_max_oper,
    )

    # 패딩값을 0으로 통일
    collate_fn = TimeSeriesOperCollate(
        continuous_padding_value=dataset_config.get("continuous_padding_value", 0.0),
        categorical_padding_value=0,  # 임베딩 레이어의 padding_idx와 일치하도록 0으로 고정
        target_padding_value=dataset_config.get("target_padding_value", 0.0),
        global_max_oper=global_max_oper,
    )

    train_dataloader = DataLoader(
        train_dataset_new,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset_new,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset_new,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"DataLoader 생성 완료:")
    logger.info(f"  - 배치 크기: {batch_size}")
    logger.info(f"  - 전체 데이터셋 global_max_oper: {global_max_oper}")
    logger.info(f"  - 범주형 패딩값: 0 (임베딩 레이어와 일치)")
    logger.info(f"  - 훈련 배치 수: {len(train_dataloader)}")
    logger.info(f"  - 검증 배치 수: {len(val_dataloader)}")
    logger.info(f"  - 테스트 배치 수: {len(test_dataloader)}")

    return train_dataloader, val_dataloader, test_dataloader
