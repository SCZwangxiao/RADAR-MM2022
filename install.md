# Testing Environments
- Linux
- Python 3.7.9
- PyTorch 1.9.0
- CUDA 10.2
- GCC 7.5.0
- mmcv-full 1.3.9

Note that the [mmcv](https://github.com/open-mmlab/mmcv) and [dgl](https://github.com/dmlc/dgl) version must be compatable with your torch and cuda version!

# Package Requirements
a. Prepare the testing environments.

b. Install additional packages.
```bash
pip install -r requirements.txt
```

# Hardware Requirements
- 60GB RAM for loading the whole dataset.

    - 60GB for Apparel dataset
    - 40GB for Food dataset
    - 20GB for Cosmetic dataset

- At least 10GB GPU memory to reproduce RADAR model. Some baselines and ablation studies requires more.