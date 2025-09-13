# #!/usr/bin/env python3
# """
# 모델 클래스들
# 임베딩 레이어, Feature Fusion, Transformer/LSTM Backbone, 전체 모델 등
# """

# import torch
# import torch.nn as nn
# from utils import logger


# class CategoricalEmbeddingLayer(nn.Module):
#     """범주형 변수들을 위한 임베딩 레이어"""

#     def __init__(self, categorical_processor, embedding_dim=8):
#         super().__init__()
#         self.categorical_processor = categorical_processor
#         self.embedding_dim = embedding_dim
#         self.embeddings = nn.ModuleList()
#         self.vocab_sizes = []

#         for col_name, encoder in categorical_processor.label_encoders.items():
#             vocab_size = len(encoder.classes_) + 1
#             self.vocab_sizes.append(vocab_size)
#             emb_layer = nn.Embedding(
#                 num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
#             )
#             self.embeddings.append(emb_layer)

#     def forward(self, categorical_data):
#         batch_size, time_window, max_oper, num_features = categorical_data.shape
#         embedded_features = []

#         for feat_idx in range(num_features):
#             feat_data = categorical_data[:, :, :, feat_idx]
#             embedded_feat = self.embeddings[feat_idx](feat_data)
#             embedded_features.append(embedded_feat)

#         if embedded_features:
#             embedded_output = torch.cat(embedded_features, dim=-1)
#         else:
#             embedded_output = torch.empty(
#                 batch_size, time_window, max_oper, 0, device=categorical_data.device
#             )

#         return embedded_output


# class FeatureFusionLayer(nn.Module):
#     """범주형 임베딩과 연속형 변수를 결합하고 1차 Linear 변환"""

#     def __init__(
#         self, continuous_dim: int, categorical_embed_dim: int, dropout: float = 0.1
#     ):
#         super().__init__()
#         self.total_feature_dim = continuous_dim + categorical_embed_dim
#         self.feature_projection = nn.Sequential(
#             nn.Linear(self.total_feature_dim, 1), nn.Dropout(dropout)
#         )

#     def forward(self, continuous_data, categorical_embedded):
#         if categorical_embedded.shape[-1] > 0:
#             combined_features = torch.cat(
#                 [continuous_data, categorical_embedded], dim=-1
#             )
#         else:
#             combined_features = continuous_data

#         projected = self.feature_projection(combined_features).squeeze(-1)
#         return projected


# class TransformerBackbone(nn.Module):
#     """Transformer Encoder 백본"""

#     def __init__(
#         self,
#         max_oper_per_hour: int,
#         time_window: int = 24,
#         num_layers: int = 6,
#         num_heads: int = 8,
#         hidden_dim: int = 512,
#         feedforward_dim: int = 2048,
#         dropout: float = 0.1,
#         activation: str = "relu",
#     ):
#         super().__init__()
#         self.hidden_dim = hidden_dim

#         self.input_projection = nn.Linear(1, hidden_dim)
#         self.time_pos_encoding = nn.Parameter(
#             torch.randn(1, time_window, 1, hidden_dim)
#         )
#         self.oper_pos_encoding = nn.Parameter(
#             torch.randn(1, 1, max_oper_per_hour, hidden_dim)
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=feedforward_dim,
#             dropout=dropout,
#             activation=activation,
#             batch_first=True,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers
#         )
#         self.output_projection = nn.Linear(hidden_dim, 1)

#     def forward(self, x, masks=None, time_masks=None):
#         batch_size, time_window, max_oper = x.shape

#         x_expanded = x.unsqueeze(-1)
#         x_projected = self.input_projection(x_expanded)

#         x_projected = x_projected + self.time_pos_encoding[:, :time_window, :, :]
#         x_projected = x_projected + self.oper_pos_encoding[:, :, :max_oper, :]

#         x_reshaped = x_projected.view(
#             batch_size, time_window * max_oper, self.hidden_dim
#         )

#         src_key_padding_mask = None
#         if masks is not None:
#             src_key_padding_mask = masks.view(batch_size, time_window * max_oper)

#         transformer_output = self.transformer_encoder(
#             x_reshaped, src_key_padding_mask=src_key_padding_mask
#         )
#         transformer_output = transformer_output.view(
#             batch_size, time_window, max_oper, self.hidden_dim
#         )

#         output = self.output_projection(transformer_output).squeeze(-1)

#         if masks is not None:
#             output = output.masked_fill(masks, 0.0)

#         return output


# class LSTMBackbone(nn.Module):
#     """LSTM 백본"""

#     def __init__(
#         self,
#         max_oper_per_hour: int,
#         time_window: int = 24,
#         num_layers: int = 2,
#         hidden_dim: int = 128,
#         dropout: float = 0.1,
#         bidirectional: bool = True,
#     ):
#         super().__init__()
#         self.hidden_dim = hidden_dim

#         self.input_projection = nn.Linear(1, hidden_dim)
#         self.lstm = nn.LSTM(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0,
#             batch_first=True,
#             bidirectional=bidirectional,
#         )

#         lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
#         self.output_projection = nn.Sequential(
#             nn.Linear(lstm_output_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, x, masks=None, time_masks=None):
#         batch_size, time_window, max_oper = x.shape

#         x_expanded = x.unsqueeze(-1)
#         x_projected = self.input_projection(x_expanded)

#         x_reshaped = x_projected.transpose(1, 2).contiguous()
#         x_reshaped = x_reshaped.view(-1, time_window, self.hidden_dim)

#         lstm_output, _ = self.lstm(x_reshaped)
#         lstm_output = lstm_output.view(batch_size, max_oper, time_window, -1)
#         lstm_output = lstm_output.transpose(1, 2)

#         output = self.output_projection(lstm_output).squeeze(-1)

#         if masks is not None:
#             output = output.masked_fill(masks, 0.0)

#         return output


# class TimeSeriesOperModel(nn.Module):
#     """완전한 시계열 공정 데이터 모델 (Feature Fusion + Backbone)"""

#     def __init__(
#         self,
#         categorical_processor,
#         continuous_dim: int,
#         max_oper_per_hour: int,
#         config: dict,
#         time_window: int = 24,
#     ):
#         super().__init__()

#         self.backbone_type = config.get("backbone_type", "transformer")
#         embedding_dim = config.get("embedding_dim", 8)
#         dropout = config.get("dropout", 0.1)

#         self.categorical_embedding = CategoricalEmbeddingLayer(
#             categorical_processor, embedding_dim
#         )

#         num_categorical_features = len(categorical_processor.label_encoders)
#         categorical_embed_dim = num_categorical_features * embedding_dim

#         self.feature_fusion = FeatureFusionLayer(
#             continuous_dim=continuous_dim,
#             categorical_embed_dim=categorical_embed_dim,
#             dropout=dropout,
#         )

#         if self.backbone_type == "transformer":
#             transformer_config = config.get("transformer", {})
#             self.backbone = TransformerBackbone(
#                 max_oper_per_hour=max_oper_per_hour,
#                 time_window=time_window,
#                 **transformer_config,
#             )
#         elif self.backbone_type == "lstm":
#             lstm_config = config.get("lstm", {})
#             self.backbone = LSTMBackbone(
#                 max_oper_per_hour=max_oper_per_hour,
#                 time_window=time_window,
#                 **lstm_config,
#             )
#         else:
#             raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

#         logger.info(f"TimeSeriesOperModel 초기화 완료:")
#         logger.info(f"  - Backbone: {self.backbone_type}")
#         logger.info(f"  - 임베딩 차원: {embedding_dim}")
#         logger.info(
#             f"  - 총 파라미터 수: {sum(p.numel() for p in self.parameters()):,}"
#         )

#     def forward(self, continuous_data, categorical_data, masks=None, time_masks=None):
#         batch_size, time_window, max_oper, _ = continuous_data.shape

#         if categorical_data.shape[-1] > 0:
#             categorical_embedded = self.categorical_embedding(categorical_data)
#         else:
#             categorical_embedded = torch.empty(
#                 batch_size, time_window, max_oper, 0, device=continuous_data.device
#             )

#         fused_output = self.feature_fusion(continuous_data, categorical_embedded)
#         predictions = self.backbone(fused_output, masks, time_masks)

#         return predictions


# def create_model_from_dataloader(dataloader, model_config: dict):
#     """DataLoader에서 정보를 추출하여 모델 생성"""

#     for batch in dataloader:
#         continuous_dim = batch["continuous_data"].shape[-1]
#         max_oper_per_hour = batch["max_oper_per_hour"]
#         time_window = batch["continuous_data"].shape[1]

#         # Dataset으로부터 categorical_processor 가져오기
#         if hasattr(dataloader.dataset, "categorical_processor"):
#             categorical_processor = dataloader.dataset.categorical_processor
#         else:
#             # Subset인 경우
#             categorical_processor = dataloader.dataset.dataset.categorical_processor

#         logger.info(f"DataLoader로부터 추출된 정보:")
#         logger.info(f"  - continuous_dim: {continuous_dim}")
#         logger.info(f"  - max_oper_per_hour: {max_oper_per_hour}")
#         logger.info(f"  - time_window: {time_window}")
#         logger.info(
#             f"  - 범주형 변수 개수: {len(categorical_processor.label_encoders)}"
#         )

#         break

#     model = TimeSeriesOperModel(
#         categorical_processor=categorical_processor,
#         continuous_dim=continuous_dim,
#         max_oper_per_hour=max_oper_per_hour,
#         config=model_config,
#         time_window=time_window,
#     )

#     return model
#!/usr/bin/env python3
"""
모델 클래스들
임베딩 레이어, Feature Fusion, Transformer/LSTM Backbone, 전체 모델 등
"""

import torch
import torch.nn as nn
from utils import logger


class CategoricalEmbeddingLayer(nn.Module):
    """범주형 변수들을 위한 임베딩 레이어"""

    def __init__(self, categorical_processor, embedding_dim=8, padding_value=0):
        super().__init__()
        self.categorical_processor = categorical_processor
        self.embedding_dim = embedding_dim
        self.padding_value = padding_value
        self.embeddings = nn.ModuleList()
        self.vocab_sizes = []

        for col_name, encoder in categorical_processor.label_encoders.items():
            # +1은 패딩을 위한 것 (인덱스 0은 패딩값으로 예약)
            vocab_size = len(encoder.classes_) + 1
            self.vocab_sizes.append(vocab_size)
            emb_layer = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0,  # 패딩 인덱스는 항상 0
            )
            self.embeddings.append(emb_layer)

        logger.info(f"임베딩 레이어 생성:")
        for i, (col_name, vocab_size) in enumerate(
            zip(categorical_processor.label_encoders.keys(), self.vocab_sizes)
        ):
            logger.info(
                f"  - {col_name}: vocab_size={vocab_size}, embedding_dim={embedding_dim}"
            )

    def forward(self, categorical_data):
        batch_size, time_window, max_oper, num_features = categorical_data.shape
        embedded_features = []

        for feat_idx in range(num_features):
            feat_data = categorical_data[:, :, :, feat_idx]

            # 인덱스 범위 검증 및 클램핑
            vocab_size = self.vocab_sizes[feat_idx]
            feat_data = torch.clamp(feat_data, 0, vocab_size - 1)

            # 디버깅을 위한 범위 체크
            if feat_data.max() >= vocab_size:
                logger.warning(
                    f"Feature {feat_idx}: max_index={feat_data.max()}, vocab_size={vocab_size}"
                )
                feat_data = torch.clamp(feat_data, 0, vocab_size - 1)

            embedded_feat = self.embeddings[feat_idx](feat_data)
            embedded_features.append(embedded_feat)

        if embedded_features:
            embedded_output = torch.cat(embedded_features, dim=-1)
        else:
            embedded_output = torch.empty(
                batch_size, time_window, max_oper, 0, device=categorical_data.device
            )

        return embedded_output


class FeatureFusionLayer(nn.Module):
    """범주형 임베딩과 연속형 변수를 결합하고 1차 Linear 변환"""

    def __init__(
        self, continuous_dim: int, categorical_embed_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.total_feature_dim = continuous_dim + categorical_embed_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(self.total_feature_dim, 1), nn.Dropout(dropout)
        )

    def forward(self, continuous_data, categorical_embedded):
        if categorical_embedded.shape[-1] > 0:
            combined_features = torch.cat(
                [continuous_data, categorical_embedded], dim=-1
            )
        else:
            combined_features = continuous_data

        projected = self.feature_projection(combined_features).squeeze(-1)
        return projected


class TransformerBackbone(nn.Module):
    """Transformer Encoder 백본"""

    def __init__(
        self,
        max_oper_per_hour: int,
        time_window: int = 24,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 512,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(1, hidden_dim)
        self.time_pos_encoding = nn.Parameter(
            torch.randn(1, time_window, 1, hidden_dim)
        )
        self.oper_pos_encoding = nn.Parameter(
            torch.randn(1, 1, max_oper_per_hour, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, x, masks=None, time_masks=None):
        batch_size, time_window, max_oper = x.shape

        x_expanded = x.unsqueeze(-1)
        x_projected = self.input_projection(x_expanded)

        x_projected = x_projected + self.time_pos_encoding[:, :time_window, :, :]
        x_projected = x_projected + self.oper_pos_encoding[:, :, :max_oper, :]

        x_reshaped = x_projected.view(
            batch_size, time_window * max_oper, self.hidden_dim
        )

        src_key_padding_mask = None
        if masks is not None:
            src_key_padding_mask = masks.view(batch_size, time_window * max_oper)

        transformer_output = self.transformer_encoder(
            x_reshaped, src_key_padding_mask=src_key_padding_mask
        )
        transformer_output = transformer_output.view(
            batch_size, time_window, max_oper, self.hidden_dim
        )

        output = self.output_projection(transformer_output).squeeze(-1)

        if masks is not None:
            output = output.masked_fill(masks, 0.0)

        return output


class LSTMBackbone(nn.Module):
    """LSTM 백본"""

    def __init__(
        self,
        max_oper_per_hour: int,
        time_window: int = 24,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, masks=None, time_masks=None):
        batch_size, time_window, max_oper = x.shape

        x_expanded = x.unsqueeze(-1)
        x_projected = self.input_projection(x_expanded)

        x_reshaped = x_projected.transpose(1, 2).contiguous()
        x_reshaped = x_reshaped.view(-1, time_window, self.hidden_dim)

        lstm_output, _ = self.lstm(x_reshaped)
        lstm_output = lstm_output.view(batch_size, max_oper, time_window, -1)
        lstm_output = lstm_output.transpose(1, 2)

        output = self.output_projection(lstm_output).squeeze(-1)

        if masks is not None:
            output = output.masked_fill(masks, 0.0)

        return output


class TimeSeriesOperModel(nn.Module):
    """완전한 시계열 공정 데이터 모델 (Feature Fusion + Backbone)"""

    def __init__(
        self,
        categorical_processor,
        continuous_dim: int,
        max_oper_per_hour: int,
        config: dict,
        time_window: int = 24,
    ):
        super().__init__()

        self.backbone_type = config.get("backbone_type", "transformer")
        embedding_dim = config.get("embedding_dim", 8)
        dropout = config.get("dropout", 0.1)

        # 패딩값을 0으로 통일
        self.categorical_embedding = CategoricalEmbeddingLayer(
            categorical_processor, embedding_dim, padding_value=0
        )

        num_categorical_features = len(categorical_processor.label_encoders)
        categorical_embed_dim = num_categorical_features * embedding_dim

        self.feature_fusion = FeatureFusionLayer(
            continuous_dim=continuous_dim,
            categorical_embed_dim=categorical_embed_dim,
            dropout=dropout,
        )

        if self.backbone_type == "transformer":
            transformer_config = config.get("transformer", {})
            self.backbone = TransformerBackbone(
                max_oper_per_hour=max_oper_per_hour,
                time_window=time_window,
                **transformer_config,
            )
        elif self.backbone_type == "lstm":
            lstm_config = config.get("lstm", {})
            self.backbone = LSTMBackbone(
                max_oper_per_hour=max_oper_per_hour,
                time_window=time_window,
                **lstm_config,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        logger.info(f"TimeSeriesOperModel 초기화 완료:")
        logger.info(f"  - Backbone: {self.backbone_type}")
        logger.info(f"  - 임베딩 차원: {embedding_dim}")
        logger.info(
            f"  - 총 파라미터 수: {sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, continuous_data, categorical_data, masks=None, time_masks=None):
        batch_size, time_window, max_oper, _ = continuous_data.shape

        if categorical_data.shape[-1] > 0:
            # 범주형 데이터의 값 범위 체크 및 클램핑
            categorical_data = torch.clamp(categorical_data, 0, 10000)  # 임시 상한값
            categorical_embedded = self.categorical_embedding(categorical_data)
        else:
            categorical_embedded = torch.empty(
                batch_size, time_window, max_oper, 0, device=continuous_data.device
            )

        fused_output = self.feature_fusion(continuous_data, categorical_embedded)
        predictions = self.backbone(fused_output, masks, time_masks)

        return predictions


def create_model_from_dataloader(dataloader, model_config: dict):
    """DataLoader에서 정보를 추출하여 모델 생성"""

    for batch in dataloader:
        continuous_dim = batch["continuous_data"].shape[-1]
        max_oper_per_hour = batch["max_oper_per_hour"]
        time_window = batch["continuous_data"].shape[1]

        # Dataset으로부터 categorical_processor 가져오기
        if hasattr(dataloader.dataset, "categorical_processor"):
            categorical_processor = dataloader.dataset.categorical_processor
        else:
            # Subset인 경우
            categorical_processor = dataloader.dataset.dataset.categorical_processor

        logger.info(f"DataLoader로부터 추출된 정보:")
        logger.info(f"  - continuous_dim: {continuous_dim}")
        logger.info(f"  - max_oper_per_hour: {max_oper_per_hour}")
        logger.info(f"  - time_window: {time_window}")
        logger.info(
            f"  - 범주형 변수 개수: {len(categorical_processor.label_encoders)}"
        )

        # 범주형 변수별 vocab_size 출력
        for col_name, encoder in categorical_processor.label_encoders.items():
            vocab_size = len(encoder.classes_) + 1
            logger.info(f"  - {col_name}: vocab_size={vocab_size}")

        break

    model = TimeSeriesOperModel(
        categorical_processor=categorical_processor,
        continuous_dim=continuous_dim,
        max_oper_per_hour=max_oper_per_hour,
        config=model_config,
        time_window=time_window,
    )

    return model
