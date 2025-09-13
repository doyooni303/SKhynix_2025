import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """기본 RNN/LSTM/GRU 모델 - 통합 임베딩 데이터셋용"""

    def __init__(
        self,
        feature_dim: int,  # 이미 결합된 특성 차원 (연속형 + 범주형 임베딩)
        rnn_type: str = "LSTM",
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        padding_value: float = -9999.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.padding_value = padding_value

        # RNN 레이어 - 입력 차원이 이미 결합된 feature_dim
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                feature_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                feature_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        else:  # RNN
            self.rnn = nn.RNN(
                feature_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

        # 출력 차원 계산
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)

        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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

        # RNN forward
        rnn_output, _ = self.rnn(masked_features)

        # 출력 레이어
        predictions = self.output_layer(rnn_output).squeeze(-1)

        # 패딩된 위치는 0으로 마스킹
        predictions = predictions.masked_fill(masks, 0.0)

        return predictions
