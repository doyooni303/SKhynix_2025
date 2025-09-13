import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


class TabularDataset(Dataset):
    def __init__(self, categorical_data, continuous_data, targets):
        self.categorical_data = torch.tensor(categorical_data, dtype=torch.long)
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "categorical": self.categorical_data[idx],
            "continuous": self.continuous_data[idx],
            "target": self.targets[idx],
        }


def sequential_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data sequentially (chronologically) maintaining order"""
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def preprocess(config):
    # Excel 파일의 모든 시트 로드 (header=1)
    data_path = config["file_path"]
    excel = pd.read_excel(data_path, sheet_name=None, header=1)

    # 지정된 시트들 결합
    sheet_names = config["sheet_names"]

    total = pd.concat([excel[sheet_name] for sheet_name in sheet_names])

    # Unnamed: 0 컬럼 제거
    if "Unnamed: 0" in total.columns:
        total.drop(columns="Unnamed: 0", inplace=True)

    # y값 결측치 제거
    original_size = len(total)
    data = total[~total[config["target_column"]].isna()]

    # x값 결측치 제거 (x22-x49 컬럼)
    drop_x_features = config.get(
        "drop_x_features", [f"x{i}" for i in range(22, 49 + 1)]
    )
    if drop_x_features:
        existing_drop_features = [col for col in drop_x_features if col in data.columns]
        if existing_drop_features:
            data = data.drop(columns=existing_drop_features, axis=1)

    # 추가 불필요한 컬럼 제거
    additional_drop_columns = config["additional_drop_columns"]
    if additional_drop_columns:
        existing_additional_drops = [
            col for col in additional_drop_columns if col in data.columns
        ]
        if existing_additional_drops:
            data = data.drop(columns=existing_additional_drops, axis=1)

    # 인덱스 리셋
    data.reset_index(drop=True, inplace=True)

    return data


def load_and_preprocess_data(config):
    """Load and preprocess the tabular data"""
    # Load data
    df = preprocess(config)

    categorical_cols = config["categorical_columns"]
    continuous_cols = config["continuous_columns"]
    target_col = config["target_column"]
    categories = config["categories"]

    print(f"Categorical columns: {categorical_cols}")
    print(f"Continuous columns: {continuous_cols}")
    print(f"Target column: {target_col}")
    print(f"Categories unique values: {categories}")

    # Handle categorical data
    categorical_data = df[categorical_cols].copy()
    label_encoders = {}

    for i, col in enumerate(categorical_cols):
        le = LabelEncoder()
        categorical_data[col] = le.fit_transform(categorical_data[col])
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values (expected: {categories[i]})")

    # Handle continuous data
    continuous_data = df[continuous_cols].copy()
    continuous_data = continuous_data.replace(np.inf, 1e5)
    scaler = None
    continuous_mean_std = None

    if config["normalize_continuous"] and len(continuous_cols) > 0:
        scaler = StandardScaler()
        continuous_data = scaler.fit_transform(continuous_data)

        # Calculate mean and std for TabTransformer
        continuous_mean_std = torch.tensor(
            [[scaler.mean_[i], scaler.scale_[i]] for i in range(len(continuous_cols))],
            dtype=torch.float32,
        )

    # Target data
    targets = df[target_col].values

    # Sequential split (8:1:1)
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    test_ratio = config["test_ratio"]

    # Prepare data for sequential split
    X_cat = categorical_data.values
    X_cont = (
        continuous_data
        if isinstance(continuous_data, np.ndarray)
        else continuous_data.values
    )

    # Sequential splits
    X_cat_train, X_cat_val, X_cat_test = sequential_split(
        X_cat, train_ratio, val_ratio, test_ratio
    )
    X_cont_train, X_cont_val, X_cont_test = sequential_split(
        X_cont, train_ratio, val_ratio, test_ratio
    )
    y_train, y_val, y_test = sequential_split(
        targets, train_ratio, val_ratio, test_ratio
    )

    # Create datasets
    train_dataset = TabularDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = TabularDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = TabularDataset(X_cat_test, X_cont_test, y_test)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "categories": categories,
        "num_continuous": len(continuous_cols),
        "continuous_mean_std": continuous_mean_std,
        "label_encoders": label_encoders,
        "scaler": scaler,
    }


class SequentialBatchSampler:
    """Custom sampler that maintains order within batches but shuffles batches"""

    def __init__(self, dataset_size, batch_size, shuffle_batches=True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches

    def __iter__(self):
        # Create sequential batches
        batches = []
        for i in range(0, self.dataset_size, self.batch_size):
            batch = list(range(i, min(i + self.batch_size, self.dataset_size)))
            batches.append(batch)

        # Shuffle the order of batches if requested
        if self.shuffle_batches:
            np.random.shuffle(batches)

        # Yield indices from each batch
        for batch in batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return self.dataset_size


def create_dataloaders(datasets, batch_size):
    """Create PyTorch DataLoaders with custom sampling"""

    # For training: shuffle batches but maintain order within batches
    train_sampler = SequentialBatchSampler(
        len(datasets["train_dataset"]), batch_size, shuffle_batches=True
    )

    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,  # Drop last incomplete batch
    )

    # For validation and test: no shuffling
    val_loader = DataLoader(
        datasets["val_dataset"], batch_size=batch_size, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        datasets["test_dataset"], batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
