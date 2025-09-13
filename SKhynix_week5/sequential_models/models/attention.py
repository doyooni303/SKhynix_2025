import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-Attention 메커니즘"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] (True = 패딩)
        """
        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(
                attention_mask, float("-inf")
            )

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )

        # Residual connection + Layer norm
        output = self.layer_norm(x + attended)

        return output


class RNNAttentionModel(nn.Module):
    """RNN + Self-Attention 모델 - 통합 임베딩 데이터셋용"""

    def __init__(
        self,
        feature_dim: int,  # 이미 결합된 특성 차원 (연속형 + 범주형 임베딩)
        rnn_type: str = "LSTM",
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
        padding_value: float = -9999.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
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

        # RNN 출력 차원
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)

        # RNN 출력을 어텐션 입력 차원으로 변환
        self.rnn_projection = nn.Linear(rnn_output_dim, hidden_dim)

        # Self-Attention
        self.self_attention = SelfAttention(hidden_dim, num_attention_heads, dropout)

        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
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

        # RNN 출력 차원 변환
        projected_output = self.rnn_projection(rnn_output)

        # Self-Attention 적용
        attended_output = self.self_attention(projected_output, masks)

        # 출력 레이어
        predictions = self.output_layer(attended_output).squeeze(-1)

        # 패딩된 위치는 0으로 마스킹
        predictions = predictions.masked_fill(masks, 0.0)

        return predictions
