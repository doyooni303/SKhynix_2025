import torch
import torch.nn as nn
import logging
from typing import Dict
from torch.utils.data import DataLoader

# 모델들 import
from .rnn import RNNModel
from .attention import RNNAttentionModel
from .transformer import TransformerModel
from .cnn import CNN1DModel

logger = logging.getLogger(__name__)


def create_model(model_config: Dict, feature_dim: int):
    """설정에 따른 모델 생성 - 통합 임베딩 데이터셋용 (모든 모델 타입 지원)"""

    model_type = model_config.get("model_type", "lstm").lower()
    dropout = model_config.get("dropout", 0.1)
    padding_value = model_config.get("padding_value", -9999.0)
    hidden_dim = model_config.get("hidden_dim", 128)
    num_layers = model_config.get("num_layers", 2)
    bidirectional = model_config.get("bidirectional", True)

    logger.info(f"모델 생성 중:")
    logger.info(f"  - 모델 타입: {model_type}")
    logger.info(f"  - 입력 특성 차원: {feature_dim}")
    logger.info(f"  - 히든 차원: {hidden_dim}")
    logger.info(f"  - 레이어 수: {num_layers}")
    logger.info(f"  - 드롭아웃: {dropout}")

    if model_type == "transformer":
        # Transformer 모델
        d_model = model_config.get("d_model", 256)
        num_heads = model_config.get("num_heads", 8)
        dim_feedforward = model_config.get("dim_feedforward", 1024)
        activation = model_config.get("activation", "relu")
        use_positional_encoding = model_config.get("use_positional_encoding", True)

        model = TransformerModel(
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_positional_encoding=use_positional_encoding,
            padding_value=padding_value,
        )

        logger.info(f"  - d_model: {d_model}")
        logger.info(f"  - num_heads: {num_heads}")
        logger.info(f"  - dim_feedforward: {dim_feedforward}")
        logger.info(
            f"  - 위치 인코딩: {'사용' if use_positional_encoding else '미사용'}"
        )

    elif model_type in ["rnn", "lstm", "gru"]:
        # 기본 RNN/LSTM/GRU 모델
        model = RNNModel(
            feature_dim=feature_dim,
            rnn_type=model_type.upper(),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            padding_value=padding_value,
        )

        logger.info(f"  - RNN 타입: {model_type.upper()}")
        logger.info(f"  - 양방향: {'Yes' if bidirectional else 'No'}")

    elif model_type in ["rnn_attention", "lstm_attention", "gru_attention"]:
        # RNN + Self-Attention 모델
        rnn_type = model_type.replace("_attention", "").upper()
        num_attention_heads = model_config.get("num_attention_heads", 8)

        model = RNNAttentionModel(
            feature_dim=feature_dim,
            rnn_type=rnn_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            bidirectional=bidirectional,
            padding_value=padding_value,
        )

        logger.info(f"  - RNN 타입: {rnn_type}")
        logger.info(f"  - 양방향: {'Yes' if bidirectional else 'No'}")
        logger.info(f"  - Attention heads: {num_attention_heads}")

    elif model_type == "cnn1d":
        # CNN 1D 모델
        kernel_sizes = model_config.get("kernel_sizes", [3, 5, 7])
        num_filters = model_config.get("num_filters", 64)

        model = CNN1DModel(
            feature_dim=feature_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            padding_value=padding_value,
        )

        logger.info(f"  - 커널 크기들: {kernel_sizes}")
        logger.info(f"  - 필터 수: {num_filters}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 파라미터 수 계산 및 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  - 총 파라미터 수: {total_params:,}")
    logger.info(f"  - 학습 가능한 파라미터 수: {trainable_params:,}")
    logger.info(f"모델 생성 완료!")

    return model


def create_model_from_config_and_dataloader(
    model_config: Dict, train_loader: DataLoader
) -> nn.Module:
    """데이터로더에서 특성 차원을 추출하여 모델 생성하는 헬퍼 함수"""

    # 첫 번째 배치에서 특성 차원 추출
    sample_batch = next(iter(train_loader))
    feature_dim = sample_batch["features"].shape[
        -1
    ]  # [batch_size, seq_len, feature_dim]

    logger.info(f"데이터로더에서 추출한 특성 차원: {feature_dim}")

    return create_model(model_config, feature_dim)
