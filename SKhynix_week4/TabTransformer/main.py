import torch
import argparse
from utils import load_config, set_random_seeds
from dataset import load_and_preprocess_data, create_dataloaders
from models import create_model
from train import train_model, evaluate_model


def run():
    parser = argparse.ArgumentParser(description="TabTransformer Training Pipeline")
    parser.add_argument(
        "--config-dir", default="configs", help="Configuration directory"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train", help="Mode: train or eval"
    )
    parser.add_argument(
        "--model-path", default=None, help="Path to saved model for evaluation"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory for saving results"
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use",
    )

    args = parser.parse_args()

    # Set random seeds
    set_random_seeds(42)

    # Load configuration
    config = load_config(args.config_dir)

    # Create experiment directory
    import os
    from datetime import datetime

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{timestamp}"
    else:
        exp_name = args.exp_name

    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_info = load_and_preprocess_data(config)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_info, config["training"]["batch_size"]
    )

    print(f"Data loaded successfully:")
    print(f"  - Categories: {data_info['categories']}")
    print(f"  - Number of continuous features: {data_info['num_continuous']}")
    print(f"  - Train samples: {len(data_info['train_dataset'])}")
    print(f"  - Validation samples: {len(data_info['val_dataset'])}")
    print(f"  - Test samples: {len(data_info['test_dataset'])}")

    # Create model
    model = create_model(
        config,
        data_info["categories"],
        data_info["num_continuous"],
        data_info["continuous_mean_std"],
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    if args.mode == "train":
        # Training mode
        print("Starting training...")

        # Update config with experiment directory for model saving
        config["exp_dir"] = exp_dir

        trained_model = train_model(model, train_loader, val_loader, config, device)

        # Load best model for final evaluation
        best_model_path = os.path.join(exp_dir, "best_model.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        print("Training completed. Evaluating on test set...")

    else:
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("--model-path must be provided in evaluation mode")

        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation on test set
    model.to(device)
    eval_metrics = (
        config["evaluation"]["metrics"] if "evaluation" in config else ["mae", "mape"]
    )
    test_metrics, predictions, targets = evaluate_model(
        model, test_loader, device, eval_metrics
    )

    print("\nTest Set Results:")
    print("=" * 50)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save predictions
    import pandas as pd
    import json
    from datetime import datetime

    # Save predictions CSV
    results_df = pd.DataFrame(
        {"actual": targets, "predicted": predictions, "residual": targets - predictions}
    )
    predictions_path = os.path.join(exp_dir, "test_predictions.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to '{predictions_path}'")

    # Save test results as JSON
    test_results = {
        "experiment_info": {
            "exp_name": exp_name,
            "exp_dir": exp_dir,
            "timestamp": datetime.now().isoformat(),
            "mode": args.mode,
            "model_path": (
                args.model_path
                if args.mode == "eval"
                else os.path.join(exp_dir, "best_model.pth")
            ),
        },
        "dataset_info": {
            "file_path": config["file_path"],
            "categories": data_info["categories"],
            "num_continuous": data_info["num_continuous"],
            "train_samples": len(data_info["train_dataset"]),
            "val_samples": len(data_info["val_dataset"]),
            "test_samples": len(data_info["test_dataset"]),
        },
        "model_info": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "test_metrics": test_metrics,
        "config": config,
    }

    results_path = os.path.join(exp_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"Test results saved to '{results_path}'")

    # Save config for reproducibility
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Configuration saved to '{config_path}'")

    print(f"\nAll outputs saved in: {exp_dir}")


if __name__ == "__main__":
    run()
