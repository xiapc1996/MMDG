import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import wave
import json
from datasets.sequence_aug import *
from datasets.MOTOR_MultiSensor import get_multimodal, Category, Label, WC

def preprocess_and_save(data_dir, output_dir, conditions):
    """
    Preprocess multimodal motor dataset and save results to disk.

    Args:
        data_dir (str): Path to the root directory containing raw data.
        output_dir (str): Directory where processed files will be written.
        conditions (list[int]): List of condition identifiers to process.

    Output files (per condition):
        - data.npz: contains arrays 'vibration', 'current', 'audio', 'labels'
        - dataset_info.npy: dictionary with metadata and shapes
        - preprocess_params.json: saved preprocessing parameters (in root output)
    """

    # Ensure the global output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Preprocessing configuration for each modality. These parameters are
    # saved to disk so experiments can be reproduced exactly.
    preprocess_params = {
        'vibration': {
            'stft': {
                'n_fft': 127,
                'hop_length': 33,
                'window': 'hann'
            }
        },
        'current': {
            'normalize': 'mean-std'
        },
        'audio': {
            # 'amplitude' indicates division by 32768.0 for 16-bit PCM audio
            'normalize': 'amplitude',
            'mel': {
                'n_fft': 512,
                'hop_length': 138,
                'n_mels': 64,
                'sr': 44100
            }
        }
    }

    # Persist the preprocessing parameters so the pipeline is auditable.
    with open(os.path.join(output_dir, 'preprocess_params.json'), 'w') as f:
        json.dump(preprocess_params, f, indent=4)

    # Build the transformation pipeline for each sensor/modal input.
    transforms = {
        'vibration': Compose([
            Reshape(),
            Retype(),
            STFT_Transform(**preprocess_params['vibration']['stft']),
        ]),
        'current': Compose([
            Reshape(),
            Retype(),
        ]),
        'audio': Compose([
            Reshape(),
            Retype(),
            Audio_Normalization(),
            MelSpectrogram_Transform(**preprocess_params['audio']['mel']),
        ])
    }

    # Process each requested working condition separately and save results
    # under `output_dir/condition_<WC[condition]>`.
    for condition in conditions:
        condition_dir = os.path.join(output_dir, f'condition_{WC[condition]}')
        os.makedirs(condition_dir, exist_ok=True)

        # Load multimodal data for the current condition. Expected format is
        # (list_of_data_dicts, list_of_labels).
        print(f"\nProcessing condition {condition} ({WC[condition]})...")
        list_data = get_multimodal(data_dir, [condition])

        print("Preprocessing data...")
        processed_data = {
            'vibration': [],
            'current': [],
            'audio': [],
            'labels': []
        }

        # Iterate through samples.
        for data, label in tqdm(zip(list_data[0], list_data[1])):
            # Apply modality-specific transforms. Each transform should return
            # a numpy array (or an object that can be stacked into one).
            vib_data = transforms['vibration'](data['vibration'])
            cur_data = transforms['current'](data['current'])
            aud_data = transforms['audio'](data['audio'])

            # Collect processed sample
            processed_data['vibration'].append(vib_data)
            processed_data['current'].append(cur_data)
            processed_data['audio'].append(aud_data)
            processed_data['labels'].append(label)

        # Stack lists into numpy arrays.
        for key in ['vibration', 'current', 'audio']:
            processed_data[key] = np.stack(processed_data[key])
        processed_data['labels'] = np.array(processed_data['labels'])

        # Save processed arrays in a single .npz archive for compactness.
        print(f"Saving data for condition {condition}...")
        np.savez(
            os.path.join(condition_dir, 'data.npz'),
            vibration=processed_data['vibration'],
            current=processed_data['current'],
            audio=processed_data['audio'],
            labels=processed_data['labels']
        )

        # Create a small metadata dictionary for downstream experiments and
        # quick inspection (number of samples, condition id/name, label maps,
        # and the shapes of the saved arrays).
        dataset_info = {
            'num_samples': len(processed_data['labels']),
            'condition': condition,
            'condition_name': WC[condition],
            'categories': Category,
            'labels': Label,
            'data_shapes': {
                'vibration': processed_data['vibration'].shape,
                'current': processed_data['current'].shape,
                'audio': processed_data['audio'].shape
            }
        }
        np.save(os.path.join(condition_dir, 'dataset_info.npy'), dataset_info)

        print(f"Condition {condition} completed. Data saved to {condition_dir}")
        print(f"Data shapes:")
        print(f"Vibration: {processed_data['vibration'].shape}")
        print(f"Current: {processed_data['current'].shape}")
        print(f"Audio: {processed_data['audio'].shape}")

def load_preprocessed_data(data_dir, condition):
    """
    Load preprocessed data and metadata for a specific working condition.

    Args:
        data_dir (str): Root directory where processed condition subfolders live.
        condition (int): Working condition identifier (used to look up WC).

    Returns:
        data (np.lib.npyio.NpzFile): Loaded .npz archive with arrays.
        dataset_info (dict): Dictionary saved alongside the arrays with metadata.
    """

    condition_dir = os.path.join(data_dir, f'condition_{WC[condition]}')

    # Load saved dataset metadata (dict) and processed arrays (.npz).
    dataset_info = np.load(os.path.join(condition_dir, 'dataset_info.npy'), allow_pickle=True).item()
    data = np.load(os.path.join(condition_dir, 'data.npz'))

    return data, dataset_info

if __name__ == "__main__":
    # Example usage when running this module directly. Adjust paths as needed.
    data_dir = "D:/xiapc_data/Motor_data"  # Path to raw/original data
    output_dir = "./Motor_processed_MMdata"  # Directory to write processed outputs

    # Example list of working condition identifiers to process.
    conditions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Run preprocessing and save outputs for each condition.
    preprocess_and_save(data_dir, output_dir, conditions)

    # Quick smoke test: load back each saved condition and print shapes.
    print("\nTesting data loading...")
    for condition in conditions:
        print(f"\nLoading condition {condition}...")
        data, dataset_info = load_preprocessed_data(output_dir, condition)
        print(f"Loaded {dataset_info['num_samples']} samples")
        print(f"Data shapes:")
        print(f"Vibration: {data['vibration'].shape}")
        print(f"Current: {data['current'].shape}")
        print(f"Audio: {data['audio'].shape}") 