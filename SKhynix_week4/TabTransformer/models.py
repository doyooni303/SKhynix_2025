import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer


def create_model(config, categories, num_continuous, continuous_mean_std=None):
    """Create TabTransformer model"""

    # Get activation function
    if config["mlp_act"] == "ReLU":
        mlp_act = nn.ReLU()
    elif config["mlp_act"] == "GELU":
        mlp_act = nn.GELU()
    elif config["mlp_act"] == "SELU":
        mlp_act = nn.SELU()
    else:
        mlp_act = nn.ReLU()

    model = TabTransformer(
        categories=tuple(categories),
        num_continuous=num_continuous,
        dim=config["dim"],
        dim_out=config["dim_out"],
        depth=config["depth"],
        heads=config["heads"],
        dim_head=config["dim_head"],
        attn_dropout=config["attn_dropout"],
        ff_dropout=config["ff_dropout"],
        mlp_hidden_mults=tuple(config["mlp_hidden_mults"]),
        mlp_act=mlp_act,
        continuous_mean_std=continuous_mean_std,
    )

    return model
