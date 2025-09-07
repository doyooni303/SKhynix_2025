import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """사인/코사인 위치 인코딩"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 위치 인코딩 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :, :].transpose(
            0, 1
        )  # [batch_size, seq_len, d_model]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Point-wise Transformer 모델 (인코더만 사용) - 통합 임베딩 데이터셋용"""

    def __init__(
        self,
        feature_dim: int,  # 입력 특성 차원 (연속형 + 범주형 임베딩 flatten 결과)
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        use_positional_encoding: bool = True,
        padding_value: float = -9999.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.padding_value = padding_value
        self.safe_padding_value = 0.0  # 🔧 안전한 패딩값 (계산용)
        self.use_positional_encoding = use_positional_encoding

        # 입력 특성을 d_model 차원으로 projection
        self.input_projection = nn.Linear(feature_dim, d_model)

        # 위치 인코딩 (선택적)
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # [batch_size, seq_len, d_model]
            norm_first=False,  # Post-norm (표준)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Point-wise 출력 레이어 (각 위치별로 독립적 예측)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # 파라미터 초기화
        self._init_parameters()

    def _init_parameters(self):
        """Xavier uniform 초기화"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, masks, **kwargs):
        """
        Args:
            features: [batch_size, seq_len, feature_dim] - 이미 결합된 특성 (연속형 + 범주형 임베딩)
            masks: [batch_size, seq_len] (True = 패딩 위치)
            **kwargs: 호환성을 위한 추가 인자들 (무시됨)

        Returns:
            predictions: [batch_size, seq_len] - 각 위치별 예측값
        """
        batch_size, seq_len, _ = features.shape

        # 🔧 핵심 수정: RNN처럼 입력에서 먼저 패딩 마스킹 수행
        # 패딩된 위치를 안전한 값으로 변경
        masked_features = features.masked_fill(
            masks.unsqueeze(-1), self.safe_padding_value
        )

        # 입력 특성을 Transformer 차원으로 projection
        x = self.input_projection(masked_features)  # 이제 안전한 값들만 처리

        # 위치 인코딩 추가 (선택적)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        # Transformer 인코더 적용
        # src_key_padding_mask: True인 위치는 attention에서 무시
        transformer_output = self.transformer_encoder(
            x, src_key_padding_mask=masks
        )  # [batch_size, seq_len, d_model]

        # Point-wise 예측 (각 위치별로 독립적)
        predictions = self.output_projection(transformer_output).squeeze(
            -1
        )  # [batch_size, seq_len]

        # 패딩된 위치는 0으로 마스킹
        predictions = predictions.masked_fill(masks, 0.0)

        return predictions
