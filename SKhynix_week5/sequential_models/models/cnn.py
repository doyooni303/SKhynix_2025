import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CNN1DModel(nn.Module):
    """1D CNN 모델 (다중 커널) - 통합 임베딩 데이터셋용"""

    def __init__(
        self,
        feature_dim: int,  # 이미 결합된 특성 차원 (연속형 + 범주형 임베딩)
        kernel_sizes: List[int] = [3, 5, 7],
        num_filters: int = 64,
        dropout: float = 0.1,
        padding_value: float = -9999.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.padding_value = padding_value

        # 다중 커널 1D Conv 레이어들 - 입력 차원이 이미 결합된 feature_dim
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    feature_dim, num_filters, kernel_size, padding=kernel_size // 2
                )
                for kernel_size in kernel_sizes
            ]
        )

        # Batch normalization
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_filters) for _ in kernel_sizes]
        )

        # 출력 레이어
        total_filters = len(kernel_sizes) * num_filters
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, masks, **kwargs):
        """
        Args:
            features: [batch_size, seq_len, feature_dim] - 이미 결합된 특성 (연속형 + 범주형 임베딩)
            masks: [batch_size, seq_len] (True = 패딩)
            **kwargs: 호환성을 위한 추가 인자들 (무시됨)

        Returns:
            predictions: [batch_size, seq_len]
        """
        batch_size, seq_len = features.shape[:2]

        # 패딩된 위치를 마스킹
        masked_features = features.masked_fill(masks.unsqueeze(-1), self.padding_value)

        # Conv1d를 위해 차원 변환: [batch, seq_len, features] -> [batch, features, seq_len]
        conv_input = masked_features.transpose(1, 2)

        # 다중 커널 Conv1D 적용
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(conv_input)))  # [batch, filters, seq_len]
            conv_outputs.append(conv_out)

        # 모든 커널 출력 결합
        combined_conv = torch.cat(
            conv_outputs, dim=1
        )  # [batch, total_filters, seq_len]

        # 다시 원래 차원으로: [batch, total_filters, seq_len] -> [batch, seq_len, total_filters]
        combined_conv = combined_conv.transpose(1, 2)

        # 출력 레이어
        predictions = self.output_layer(combined_conv).squeeze(-1)

        # 패딩된 위치는 0으로 마스킹
        predictions = predictions.masked_fill(masks, 0.0)

        return predictions
