# HCL-Ms3AD

Tri-modal anomaly detection project for MulSen-AD style RGB, infrared, and point-cloud data. The code supports single-modality baselines, gated multi-modal fusion, and a fusion-memory workflow with cached frozen features and global high-order contrastive fusion pretraining.

## Project Structure

```text
.
├── Test.py                  # Main entry point
├── runner.py                # Training, feature caching, memory bank, and evaluation runner
├── dataset.py               # MulSen-AD dataset loader
├── dataset_3D.py            # 3D dataset utilities
├── feature_extractors/      # RGB, infrared, point-cloud, and fusion feature extractors
├── models/                  # Backbone and fusion modules
└── utils_loc/               # Metrics and utility functions
```

## Requirements

This project depends on Python deep-learning and 3D processing packages, including:

- PyTorch
- torchvision
- timm
- numpy
- pandas
- scipy
- scikit-learn
- tqdm
- Pillow
- matplotlib
- OpenCV
- open3d
- trimesh
- tifffile
- pointnet2_ops
- knn_cuda

Install the versions that match your CUDA and PyTorch environment. For example:

```bash
pip install torch torchvision timm numpy pandas scipy scikit-learn tqdm pillow matplotlib opencv-python open3d trimesh tifffile
```

`pointnet2_ops` and `knn_cuda` usually require CUDA-specific installation steps.

## Dataset

By default, the code expects the dataset at:

```text
./dataset/MulSen_AD
```

Each class should contain RGB, infrared, and point-cloud folders in the format expected by `dataset.py`, for example:

```text
MulSen_AD/
└── capsule/
    ├── RGB/
    ├── Infrared/
    └── Pointcloud/
```

You can override the dataset path with `--dataset_path`.

## Usage

Run all classes with the default method:

```bash
python Test.py --dataset_path ./dataset/MulSen_AD
```

Run selected classes:

```bash
python Test.py --classes capsule,cotton,cube --dataset_path ./dataset/MulSen_AD
```

Run the fusion-memory workflow in three stages:

```bash
python Test.py --method_name PC+RGB+Infra+fusion_memory --fusion_extract_only --dataset_path ./dataset/MulSen_AD
python Test.py --method_name PC+RGB+Infra+fusion_memory --fusion_pretrain_only --dataset_path ./dataset/MulSen_AD
python Test.py --method_name PC+RGB+Infra+fusion_memory --dataset_path ./dataset/MulSen_AD
```

Outputs are saved under `output_dir/` by default.
