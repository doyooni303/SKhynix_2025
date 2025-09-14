import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List


"""Encoding Modules"""


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class CategoricalEmbedding(nn.Module):
    """범주형 변수를 위한 임베딩 모듈"""

    def __init__(self, vocab_sizes: List[int], embedding_dim: int):
        super(CategoricalEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim) for vocab_size in vocab_sizes]
        )

    def forward(self, categorical_inputs):
        """
        Args:
            categorical_inputs: [batch_size, seq_len, num_categorical_features]
        Returns:
            embedded: [batch_size, seq_len, num_categorical_features * embedding_dim]
        """
        embedded_features = []
        for i, embedding_layer in enumerate(self.embeddings):
            # 각 범주형 변수에 대해 임베딩 수행
            cat_input = categorical_inputs[:, :, i].long()
            embedded = embedding_layer(cat_input)
            embedded_features.append(embedded)

        # 모든 임베딩을 concatenate
        return torch.cat(embedded_features, dim=-1)


class DataEmbeddingWithCategorical(nn.Module):
    """범주형 및 연속형 변수를 모두 처리하는 임베딩 모듈 (특별 토큰 포함)"""

    def __init__(
        self,
        continuous_dim: int,
        vocab_sizes: List[int],
        d_model: int,
        window_size: int = 10,
        embedding_dim: int = 8,
        dropout: float = 0.1,
        use_special_tokens: bool = False,
        num_groups: int = None,
    ):
        super(DataEmbeddingWithCategorical, self).__init__()

        self.window_size = window_size
        self.use_special_tokens = use_special_tokens
        self.d_model = d_model

        # 범주형 변수 임베딩
        self.categorical_embedding = CategoricalEmbedding(vocab_sizes, embedding_dim)

        # 범주형 임베딩과 연속형 변수를 결합한 차원
        total_input_dim = continuous_dim + len(vocab_sizes) * embedding_dim

        # 입력을 d_model 차원으로 투영
        self.input_projection = nn.Linear(total_input_dim, d_model)

        # 특별 토큰 (learnable parameters)
        if use_special_tokens:
            self.seq_token = nn.Parameter(torch.randn(1, 1, d_model))

            if num_groups is not None:
                self.group_tokens = nn.Parameter(torch.randn(num_groups, 1, d_model))
            else:
                self.group_token_embedding = nn.Embedding(1000, d_model)

        # Positional 임베딩 (Window 크기에 맞춤)
        self.position_embedding = PositionalEmbedding(d_model, max_len=window_size)

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, continuous_x, categorical_x, position_ids=None, reference_groups=None
    ):
        batch_size, seq_len = continuous_x.shape[:2]

        # 범주형 변수 임베딩
        cat_embedded = self.categorical_embedding(categorical_x)

        # 연속형과 범주형 결합
        combined = torch.cat([continuous_x, cat_embedded], dim=-1)

        # d_model 차원으로 투영
        x = self.input_projection(combined)

        # Window 부분에만 positional embedding 적용
        if position_ids is not None:
            for b in range(batch_size):
                window_mask = position_ids[b] == 0
                window_indices = window_mask.nonzero(as_tuple=False).squeeze(-1)

                if len(window_indices) > 0:
                    window_length = len(window_indices)
                    pos_emb = self.position_embedding.pe[:, :window_length, :].squeeze(
                        0
                    )
                    x[b, window_indices] = x[b, window_indices] + pos_emb
        else:
            if seq_len <= self.window_size:
                x = x + self.position_embedding(x)
            else:
                pos_encoding = self.position_embedding.pe[:, : self.window_size, :]
                x[:, : self.window_size, :] = x[:, : self.window_size, :] + pos_encoding

        # 특별 토큰 추가
        if (
            self.use_special_tokens
            and position_ids is not None
            and reference_groups is not None
        ):
            new_x = []
            new_masks = []

            for b in range(batch_size):
                # 각 부분 추출
                padding_mask = position_ids[b] == -1
                window_mask = position_ids[b] == 0
                group_mask = position_ids[b] == 1

                window_data = x[b, window_mask]
                group_data = x[b, group_mask]
                padding_data = x[b, padding_mask]

                # 새로운 시퀀스 구성
                seq_parts = []
                mask_parts = []

                # 1. [SEQ] 토큰
                seq_token = self.seq_token.view(1, -1)
                seq_parts.append(seq_token)
                mask_parts.append(torch.ones(1, device=x.device, dtype=torch.bool))

                # 2. Window 데이터
                if window_data.numel() > 0:
                    if window_data.dim() == 1:
                        window_data = window_data.unsqueeze(0)
                    seq_parts.append(window_data)
                    mask_parts.append(
                        torch.ones(
                            window_data.shape[0], device=x.device, dtype=torch.bool
                        )
                    )

                # 3. [GROUP] 토큰
                group_id = (
                    reference_groups[b].item()
                    if hasattr(reference_groups[b], "item")
                    else reference_groups[b]
                )
                if hasattr(self, "group_tokens"):
                    group_token = self.group_tokens[group_id].view(1, -1)
                else:
                    group_token = self.group_token_embedding(
                        torch.tensor([group_id], device=x.device)
                    ).view(1, -1)
                seq_parts.append(group_token)
                mask_parts.append(torch.ones(1, device=x.device, dtype=torch.bool))

                # 4. Group 데이터
                if group_data.numel() > 0:
                    if group_data.dim() == 1:
                        group_data = group_data.unsqueeze(0)
                    seq_parts.append(group_data)
                    mask_parts.append(
                        torch.ones(
                            group_data.shape[0], device=x.device, dtype=torch.bool
                        )
                    )

                # 5. 패딩 데이터 (마스크는 False)
                if padding_data.numel() > 0:
                    if padding_data.dim() == 1:
                        padding_data = padding_data.unsqueeze(0)
                    seq_parts.append(padding_data)
                    mask_parts.append(
                        torch.zeros(
                            padding_data.shape[0], device=x.device, dtype=torch.bool
                        )
                    )  # 패딩은 False

                # 결합
                new_seq = torch.cat(seq_parts, dim=0)
                new_mask = torch.cat(mask_parts, dim=0)

                new_x.append(new_seq)
                new_masks.append(new_mask)

            # 모든 시퀀스가 같은 길이이므로 추가 패딩 불필요
            # 바로 스택 가능
            x = torch.stack(new_x)  # [batch_size, seq_len + 2, d_model]
            masks = torch.stack(new_masks)  # [batch_size, seq_len + 2]

            # Normalization과 Dropout
            x = self.norm(self.dropout(x))
            return x, masks

        # 특별 토큰 없이
        x = self.norm(self.dropout(x))
        masks = (
            (position_ids != -1)
            if position_ids is not None
            else torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)
        )
        return x, masks


"""Encoder/Decoder Modules"""


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


"""Attention Modules"""


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


"""Model"""


class VanillaTransformer(nn.Module):
    """사용자 데이터 구조에 맞춘 수정된 Vanilla Transformer"""

    def __init__(self, configs):
        super(VanillaTransformer, self).__init__()

        self.ref_idx = configs.ref_idx  # 그룹 참조 인덱스 (window 내 위치)
        self.output_type = configs.output_type  # 'sequence' or 'target'
        self.use_special_tokens = configs.use_special_tokens
        self.window_size = configs.window_size

        # 데이터 임베딩 (범주형 + 연속형)
        self.data_embedding = DataEmbeddingWithCategorical(
            continuous_dim=configs.continuous_dim,
            vocab_sizes=configs.vocab_sizes,
            d_model=configs.d_model,
            embedding_dim=configs.embedding_dim,
            dropout=configs.dropout,
            window_size=configs.window_size,
            use_special_tokens=configs.use_special_tokens,
            num_groups=configs.num_groups,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),  # 단일 값 예측
        )

    def get_inputs(self, batch, device):
        continuous_data = batch["continuous_data"].to(device)  # 연속형 데이터
        categorical_data = batch["categorical_data"].to(device)  # 범주형 데이터
        position_ids = batch["position_ids"].to(device)  # 위치 ID (window/group 구분)
        reference_oper_groups = batch["reference_oper_groups"]  # 참조 그룹
        targets = batch["targets"].to(device)  # 타겟값
        masks = batch["masks"].to(device)  # 유효 데이터 마스크
        sequence_lengths = batch["sequence_lengths"]  # 시퀀스 길이

        # 메타 정보 추출
        timekeys = batch.get("timekeys", [])  # 시간 키
        window_oper_ids = batch.get("oper_ids_list", [])  # 윈도우 공정 ID

        enc_out, new_masks = self.data_embedding(
            continuous_data, categorical_data, position_ids, reference_oper_groups
        )

        # Attention mask 처리 (False = 마스킹할 위치)
        attn_mask = None
        if new_masks is not None:
            batch_size, seq_len = new_masks.shape
            # attention에서는 True가 마스킹 위치이므로 반전
            attn_mask = ~new_masks
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            attn_mask = attn_mask.expand(-1, 1, seq_len, -1)  # [B, 1, L, L]

        return (
            enc_out,
            attn_mask,
            targets,
            new_masks,
            timekeys,
            window_oper_ids,
            sequence_lengths,
            reference_oper_groups,
        )

    def forward(self, enc_out, attn_mask):
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=attn_mask)

        if self.use_special_tokens:
            # 각 부분 추출
            seq_token = enc_out[:, 0, :]  # [B, d_model] - [SEQ] 토큰
            window_repr = enc_out[
                :, 1 : self.window_size + 1, :
            ]  # [B, window_size, d_model]
            group_token = enc_out[
                :, self.window_size + 1, :
            ]  # [B, d_model] - [GROUP] 토큰

            if self.output_type == "sequence":
                # Window 부분 전체에 대해 예측
                predictions = self.prediction_head(window_repr).squeeze(
                    -1
                )  # [B, window_size]
                # targets도 window 부분만
                # targets_out = targets[:, : self.window_size]

            elif self.output_type == "target":
                # ref_idx 위치만 사용
                ref_repr = window_repr[:, self.ref_idx, :]  # [B, d_model]
                predictions = self.prediction_head(ref_repr).squeeze(-1)  # [B]

                # # targets도 ref_idx 위치만 추출
                # targets_out = targets[:, self.ref_idx]  # [B]

        else:
            # 특별 토큰 없는 경우
            if self.output_type == "sequence":
                # 전체 시퀀스에 대해 예측
                predictions = self.prediction_head(enc_out).squeeze(-1)  # [B, seq_len]
                # targets_out = targets

            elif self.output_type == "target":
                # ref_idx 위치만 사용
                ref_repr = enc_out[:, self.ref_idx, :]  # [B, d_model]
                predictions = self.prediction_head(ref_repr).squeeze(-1)  # [B]

                # targets도 ref_idx 위치만
                # targets_out = targets[:, self.ref_idx]  # [B]

        return predictions  # , targets_out
