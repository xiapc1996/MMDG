import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

class PreprocessedDataset(Dataset):
    """
    Dataset wrapper for preprocessed sensor data saved per operating condition.

    Each condition is expected in a subfolder named `condition_{condition}` and
    to contain `data.npz` (with arrays 'vibration', 'current', 'audio', 'labels')
    and `dataset_info.npy` (meta information such as label names/categories).

    Args:
        data_dir (str): Base directory containing condition folders.
        condition (str): Name of the condition (e.g., 'MS20_LS').
        transform (callable, optional): Optional transform applied to each sample.
        samples_per_label (int|None): If set, limits the number of samples per label.
        random_seed (int): Seed used when sampling per-label subsets.
    """

    def __init__(self, data_dir: str, condition: str, transform=None, samples_per_label=None, random_seed: int = 42):
        self.data_dir = os.path.join(data_dir, f'condition_{condition}')
        self.transform = transform

        # Load dataset metadata (label names, categories, etc.)
        self.dataset_info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        self.labels = self.dataset_info['labels']
        self.categories = self.dataset_info['categories']
        
        # Load processed arrays
        data = np.load(os.path.join(self.data_dir, 'data.npz'))
        self.vibration = data['vibration']
        self.current = data['current']
        self.audio = data['audio']
        self.data_labels = data['labels']

        # If requested, select a fixed number of samples per label (stratified sampling)
        if samples_per_label is not None:
            np.random.seed(random_seed)
            selected_indices = []
            for label in np.unique(self.data_labels):
                label_indices = np.where(self.data_labels == label)[0]
                if len(label_indices) > samples_per_label:
                    selected_indices.extend(
                        np.random.choice(label_indices, samples_per_label, replace=False)
                    )
                else:
                    selected_indices.extend(label_indices)

            selected_indices = np.array(selected_indices)
            self.vibration = self.vibration[selected_indices]
            self.current = self.current[selected_indices]
            self.audio = self.audio[selected_indices]
            self.data_labels = self.data_labels[selected_indices]
        
        self.num_samples = len(self.data_labels)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return a single sample as a dict of tensors and an integer label.
        data = {
            'vibration': torch.from_numpy(self.vibration[idx]),
            'current': torch.from_numpy(self.current[idx]),
            'audio': torch.from_numpy(self.audio[idx])
        }
        label = self.data_labels[idx]
        
        # Apply optional transforms (if any)
        if self.transform is not None:
            data = self.transform(data)
        
        return data, label

class PreprocessedMultiConditionDataset(Dataset):
    """
    Concatenate multiple `PreprocessedDataset` instances (one per operating condition)
    to expose a single dataset spanning all conditions.

    This avoids copying data by delegating __getitem__ to the appropriate
    per-condition dataset based on precomputed offsets.
    """
    def __init__(self, data_dir, conditions, transform=None, samples_per_label=None, random_seed=42):
        self.data_dir = data_dir
        self.transform = transform
        self.datasets = []
        self.condition_offsets = [0]

        # Load datasets for all requested conditions and compute offsets
        total_samples = 0
        for condition in conditions:
            dataset = PreprocessedDataset(
                data_dir,
                condition,
                transform,
                samples_per_label=samples_per_label,
                random_seed=random_seed,
            )
            self.datasets.append(dataset)
            total_samples += len(dataset)
            self.condition_offsets.append(total_samples)
        
        self.num_samples = total_samples
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Identify which underlying dataset contains this global index using offsets
        dataset_idx = 0
        while dataset_idx < len(self.condition_offsets) - 1 and idx >= self.condition_offsets[dataset_idx + 1]:
            dataset_idx += 1

        local_idx = idx - self.condition_offsets[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

def create_train_val_datasets(data_dir, conditions, transform=None, samples_per_label=None, 
                            val_ratio=0.2, random_seed=42):
    """
    Create train and validation subsets from multiple conditions using stratified splitting by label.

    Returns:
        (train_dataset, val_dataset): two torch.utils.data.Subset objects
    """
    full_dataset = PreprocessedMultiConditionDataset(
        data_dir=data_dir,
        conditions=conditions,
        transform=transform,
        samples_per_label=samples_per_label,
        random_seed=random_seed,
    )
    
    # Collect labels efficiently from underlying datasets' numpy arrays
    all_labels = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        all_labels.append(label)
    all_labels = np.array(all_labels)
    
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=all_labels,
        random_state=random_seed,
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    return train_dataset, val_dataset