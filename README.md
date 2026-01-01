# MMDG

This repository contains implementation for multi-modal domain generalization (MMDG) method entitled "Multi-modal cross-domain mixed fusion model with dual disentanglement for fault diagnosis under unseen working conditions" 
* **(Jan 1, 2026)** Preprint version: [ArXiv](https://doi.org/10.48550/arXiv.2512.24679). 

The complete source code with detailed network configuration will be released shortly after the manuscript review process.

## Usage

Requirements: Python 3.8+ and a virtual environment.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Preprocess example:

```powershell
python preprocess_data.py --input-dir ./raw_data --output-dir ./datasets/preprocessed
```

Train example:

```powershell
python train_mix.py
```

Inference example:

```powershell
python main_mix.py --model-path ./checkpoints/best.pth --input ./datasets/preprocessed/test
```

## Key files

- `preprocess_data.py` — data preprocessing and saving
- `train_mix.py` — training loop and validation
- `main_mix.py` — inference and runner
- `utils.py` — helpers (metrics, logging, etc.)
- `datasets/` — data loaders and augmentation utilities
- `models/` — model implementations

## Citation

Please cite our work if you find it useful — we truly appreciate it!
```powershell
@misc{xia2025multimodalcrossdomainmixedfusion,
      title={Multi-modal cross-domain mixed fusion model with dual disentanglement for fault diagnosis under unseen working conditions}, 
      author={Pengcheng Xia and Yixiang Huang and Chengjin Qin and Chengliang Liu},
      year={2025},
      doi={10.48550/arXiv.2512.24679}
}

```
