# MMDG

This repository contains implementation for multi-modal domain generalization (MMDG) method entitled

**Multi-modal cross-domain mixed fusion model with dual disentanglement for fault diagnosis under unseen working conditions**

* **(Jul 18, 2026)** Published version: [MSSP](https://doi.org/10.1016/j.ymssp.2026.114693). 
* **(Jan 1, 2026)** Preprint version: [ArXiv](https://doi.org/10.48550/arXiv.2512.24679). 

## Requirements

Python 3.8+ is recommended. Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Data Preparation

The preprocessing script converts raw multi-modal motor data into per-condition `.npz` files. Please edit `data_dir` in `preprocess_data.py` to point to your raw dataset root if needed.

```powershell
python preprocess_data.py
```

## Usage

Run the main experiment entry:

```powershell
python main_mix.py
```

By default, the model trains on source working conditions `[1, 2, 3]` and tests on target working condition `[0]`. 

Important arguments:

- `--data_dir`: directory containing preprocessed condition folders.
- `--source_condition`: source working-condition IDs separated by spaces.
- `--target_condition`: target working-condition IDs separated by spaces.
- `--epochs`: number of training epochs.
- `--batch_size`: mini-batch size.
- `--learning_rate`: learning rate.
- `--experiment_root`: directory for logs and experiment outputs.

## Repository Structure

- `main_mix.py`: main entry for training and testing.
- `train_mix.py`: training, validation, testing, and loss functions.
- `preprocess_data.py`: raw-data preprocessing and saving.
- `datasets/`: dataset loaders and signal/audio transforms.
- `models/`: model definitions, including modality encoders, fusion module, and classifier.
- `utils.py`: metrics, logging, and helper functions.

## Models

The `models/` directory contains the modules required by the default `MultiModal` experiment:

- `MultiModal.py`: vibration, current, and audio feature encoders.
- `Resnet1d.py`: 1D ResNet backbone for current signals.
- `Resnet2d.py`: 2D ResNet backbone for vibration/audio time-frequency inputs.
- `DomainFeatureExtractor.py`: cross-attention-based triple-modal fusion.
- `FinalClassifier.py`: final fault classifier.

## Citation

Please cite our work if you find this repository useful:

```bibtex
@article{xia2026multimodal,
   author = {Xia, Pengcheng and Huang, Yixiang and Qin, Chengjin and Liu, Chengliang},
   title = {Multi-modal cross-domain mixed fusion model with dual disentanglement for fault diagnosis under unseen working conditions},
   journal = {Mechanical Systems and Signal Processing},
   volume = {258},
   DOI = {10.1016/j.ymssp.2026.114693},
   year = {2026}
}
```
