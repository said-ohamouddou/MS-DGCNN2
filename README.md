# MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification

## Overview

MS-DGCNN++ is a hierarchical multiscale fusion dynamic graph convolutional network designed for tree species classification from terrestrial LiDAR point clouds. Unlike existing approaches that use parallel multi-scale processing, our method employs semantically meaningful feature extraction at local, branch, and canopy scales with cross-scale information propagation.

media/pg_0001.png
media/pg_0002.png
media/pg_0003.png

## Key Features

- **Hierarchical Multi-Scale Processing**: Semantic feature extraction at local, branch, and canopy scales
- **Biological Knowledge Integration**: Architecture aligned with natural tree structure
- **Cross-Scale Information Propagation**: Enhanced feature fusion across different scales
- **Scale-Specific Feature Engineering**: 
  - Standard geometric features for local scale
  - Normalized relative vectors for branch scale
  - Distance information for canopy scale

## Environment Requirements

This implementation is tested on:
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.8+
- **PyTorch**: 2.4.1
- **CUDA**: 11.8

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MS-DGCNN-plus-plus.git
cd MS-DGCNN-plus-plus
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Custom Operations for Baseline MS-DGCNN

For the baseline MS-DGCNN functionality, you need to install additional modules for Farthest Point Sampling (FPS) from the Pointnet++ custom operations:

```bash
# Install custom ops from Pointnet2_PyTorch
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -e .
```

For more information, visit: [Pointnet2_PyTorch Repository](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master)

## Dataset

### STPCTLS Data

The current repository uses the STPCTLS dataset. Preprocessed data is available in H5 format in the `data_tree` folder.

<div align="center">
<img src="media/stpctls-samples.png" alt="STPCTLS Samples" width="800"/>
</div>

### Data Information

```bash
# Get dataset information
python data.py
```

### Data Visualization

```bash
# Visualize the dataset
python tree_visualizer.py
```

## Usage

### Quick Test with Pretrained Model

Test the model using pretrained weights with preprocessed data:

```bash
python test.py
```

### Training

Train the MS-DGCNN++ model:

```bash
python train.py --model MS_DGCNN2
```

### Data Preprocessing

To preprocess STPCTLS data yourself:

1. Download the raw data from the official source: https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/FOHUJM
2. Organize data files (xyz, pts, txt) of each class in separate folders named by class
3. Place all class folders in the `data_tree` directory
4. Run data.py to prepare h5 files

```bash
# Structure should be:
# data_tree/
# ├── class1/
# │   ├── sample1.xyz
# │   ├── sample1.pts
# │   └── sample1.txt
# └── class2/
#     ├── sample2.xyz
#     ├── sample2.pts
#     └── sample2.txt
```

Using this structure, the script could be adapted to any other 3D point cloud classification dataset.

## Results

### STPCTLS Dataset
#### MS-DGCNN++ performance with different k-NN configurations

| \textbf{Configuration}        | \textbf{OA}    | \textbf{\small$\kappa$} | \textbf{BA}    | \textbf{Douglas fir} | \textbf{Beech} | \textbf{Spruce} | \textbf{Red Oak} | \textbf{Oak} | \textbf{Ash}    | \textbf{Pine} | \textbf{PC} | \textbf{ ET}  |
|-------------------------------|----------------|-------------------------|----------------|----------------------|----------------|-----------------|------------------|--------------|-----------------|---------------|-------------|---------------|
| MS-DGCNN++(5,20,100)          | 94.24          | 92.73                   | 89.73          | 94.44                | \textbf{97.06} | 93.75           | 86.96            | \textbf{100} | \textbf{100.00} | \textbf{100}  | 1.81        | 6.93          |
| \textbf{MS-DGCNN++(5,20,30) } | \textbf{94.96} | \textbf{ 93.63}         | 87.40          | \textbf{100.00}      | 94.12          | \textbf{96.97}  | 86.96            | \textbf{100} | 83.33           | \textbf{100}  | 1.81        | 5.76          |
| MS-DGCNN++(5,20,25)           | 94.24          | 92.74                   | 87.80          | \textbf{100.00}      | 94.29          | \textbf{96.97}  | 90.00            | \textbf{100} | 75.00           | 80            | 1.81        | 5.82          |
| MS-DGCNN++(5,6,50)            | 89.21          | 86.44                   | 82.53          | \textbf{100.00}      | 89.19          | 90.91           | 88.89            | \textbf{100} | 58.33           | 80            | 1.81        | \textbf{3.60} |
| MS-DGCNN++(15,20,50)          | 92.81          | 90.95                   | 89.30          | 94.44                | 96.97          | 93.75           | \textbf{95.00}   | 80           | 87.50           | 60            | 1.81        | 6.34          |
| MS-DGCNN++(4,20,50)           | 93.53          | 91.83                   | 87.63          | 97.22                | 94.12          | 93.94           | \textbf{95.00}   | \textbf{100} | 75.00           | 80            | 1.81        | 6.10          |
| \textbf{MS-DGCNN++(2,20,50)}  | \textbf{94.24} | 92.73                   | \textbf{91.61} | 97.14                | 94.29          | 91.18           | 94.74            | 100          | 87.50           | \textbf{100}  | 1.81        | 5.97          |



#### Comparison with other 3D point cloud models 

| \textbf{Model}           | \textbf{OA}    | \textbf{\small$\kappa$} | \textbf{BA}    | \textbf{Douglas fir} | \textbf{Beech} | \textbf{Spruce} | \textbf{Red Oak} | \textbf{Oak}    | \textbf{Ash}    | \textbf{Pine}   | \textbf{PC}    | \textbf{ET}   |
|--------------------------|----------------|-------------------------|----------------|----------------------|----------------|-----------------|------------------|-----------------|-----------------|-----------------|----------------|---------------|
| PointNet                 | 79.86          | 74.67                   | 75.86          | 93.33                | 82.86          | 77.78           | 68.18            | 75.00           | 50.00           | \textbf{100.00} | \textbf{0.68}  | \textbf{0.77} |
| PointNet++ SSG           | 80.58          | 75.17                   | 68.81          | 70.00                | 85.71          | 90.00           | 95.24            | \textbf{100.00} | 0.00            | 57.14           | 1.46           | 3.25          |
| PointNet++ MSG           | 78.42          | 72.21                   | 68.57          | 76.09                | 67.44          | 92.31           | 87.50            | \textbf{100.00} | 50.00           | \textbf{100.00} | 1.73           | 9.40          |
| PointMLP                 | 87.77          | 84.55                   | 83.55          | 89.74                | 90.91          | 90.00           | 77.27            | \textbf{100.00} | 85.71           | 80.00           | 13.23          | 13.36         |
| PointMLP Lite            | 87.05          | 83.65                   | 83.75          | 83.72                | 88.57          | 96.00           | 94.44            | 66.67           | 75.00           | 75.00           | 0.71           | 3.64          |
| PointWeb                 | 79.86          | 74.66                   | 70.74          | 86.49                | 87.10          | 79.41           | 83.33            | 25.00           | 50.00           | 80.00           | 0.78           | 25.92         |
| CurevNet                 | 87.05          | 83.67                   | 80.30          | 96.67                | 86.84          | 83.33           | 89.47            | \textbf{100.00} | 60.00           | \textbf{100.00} | 2.12           | 5.18          |
| PointConv                | 88.49          | 85.42                   | 78.18          | 97.22                | 93.94          | 83.33           | 80.00            | 66.67           | 75.00           | \textbf{100.00} | 19.56          | 8.05          |
| PoinTnT                  | 50.36          | 38.15                   | 41.68          | 55.56                | 67.65          | 0.00            | 88.89            | 14.29           | 8.33            | 0.00            | 3.93           | 3.57          |
| GDANet                   | 89.21          | 86.42                   | 82.44          | 96.97                | 94.29          | 85.29           | 90.00            | 75.00           | 66.67           | 75.00           | 0.93           | 7.60          |
| DeepGCN                  | 89.21          | 86.46                   | 84.60          | 94.12                | 96.97          | 88.24           | 94.44            | \textbf{100.00} | 54.55           | 66.67           | 2.21           | 7.99          |
| PVT                      | 84.17          | 80.06                   | 85.19          | 78.26                | 90.91          | 91.30           | 93.75            | 80.00           | 58.33           | \textbf{100.00} | 9.16           | 17.47         |
| PCT                      | 66.91          | 58.09                   | 52.26          | 90.32                | 75.76          | 62.16           | 43.33            | 50.00           | 50.00           | 50.00           | 2.87           | 3.83          |
| PCP-MAE                  | 90.65          | 88.18                   | 89.09          | 89.19                | 91.43          | 93.33           | 82.61            | \textbf{100.00} | \textbf{100.00} | \textbf{100.00} | 22.34          | 1.77          |
| PointBert                | 92.81          | 90.99                   | 86.29          | \textbf{100.00}      | 96.77          | \textbf{100.00} | 86.96            | 75.00           | 70.00           | 60.00           | 22.06          | 4.54          |
| PointGPT-S               | 93.53          | 91.81                   | 90.92          | 90.24                | 93.94          | 96.55           | 95.24            | 80.00           | \textbf{100.00} | \textbf{100.00} | 29.23          | 2.90          |
| PPT                      | \textbf{94.24} | \textbf{92.71}          | \textbf{92.26} | 97.22                | 89.19          | 93.75           | 95.00            | \textbf{100.00} | \textbf{100.00} | \textbf{100.00} | \textbf{22.78} | \textbf{2.79} |
| ReCON                    | 91.37          | 89.12                   | 88.40          | 94.44                | 91.67          | 93.10           | 86.96            | 80.00           | \textbf{100.00} | 80.00           | 43.57          | 2.07          |
| DGCNN(50)                | 87.05          | 83.65                   | 82.54          | 87.50                | 88.89          | 92.86           | 88.24            | \textbf{100.00} | 66.67           | 66.67           | 1.80           | 11.17         |
| DGCNN(20)                | 90.65          | 88.25                   | 86.77          | 97.06                | 94.12          | 90.91           | 89.47            | \textbf{100.00} | 63.64           | 80.00           | 1.80           | 5.20          |
| DGCNN(5)                 | 89.93          | 87.33                   | 80.84          | 94.12                | \textbf{97.06} | 91.18           | 89.47            | 66.67           | 77.78           | 50.00           | 1.80           | 2.36          |
| MS-DGCNN(5,20,50)        | 87.05          | 83.67                   | 81.41          | 94.12                | 93.75          | 82.05           | 88.24            | 60.00           | 71.43           | 80.00           | 1.55           | 6.09          |
| MS-DGCNN(20,30,40)       | 86.33          | 82.64                   | 76.94          | 89.19                | 82.50          | 90.00           | 84.21            | 66.67           | 85.71           | \textbf{100.00} | 1.55           | 5.31          |
| \textbf{MS-DGCNN++(our)} | \textbf{93.53} | \textbf{91.88}          | \textbf{90.79} | \textbf{100.00}      | 96.77          | 94.12           | \textbf{100.00}  | 80.00           | 60.00           | 80.00           | \textbf{1.81}  | \textbf{6.03} |


| \textbf{Method}                  | \textbf{OA} | \textbf{\small$\kappa$} | \textbf{Douglas fir} | \textbf{Beech} | \textbf{Spruce} | \textbf{Red Oak} | \textbf{Oak} | \textbf{Ash} | \textbf{Pine} | \textbf{ET} |
|----------------------------------|-------------|-------------------------|----------------------|----------------|-----------------|------------------|--------------|--------------|---------------|-------------|
| LeNet 5 (2D projections)         | 86.01       | -                       | 93.00                | 94.00          | 84.00           | 63.00            | 82.00        | 77.00        | 92.00         | -           |
| MFFTC-Net                        | 90.37       | 87.89                   | 97.14                | 87.10          | 93.75           | 94.74            | 100.00       | 66.67        | 71.43         | 91.71       |
| MFFTC-Net (without augmentation) | 81.48       | 76.40                   | 89.47                | 83.33          | 86.32           | 73.68            | 50.00        | 80.00        | 50.00         | -           |

### XFOR-Species20K Dataset



### ModelNet10/40 Dataset
For general 3D point cloud classification benchmarks, MS-DGCNN++ outperforms DGCNN and MS-DGCNN, and achieves competitive results with transformer-based models.

## Project Structure

```
MS-DGCNN-plus-plus/
├── data_tree/              # Dataset directory
├── media/                  # Images and visualization
├── models/                 # Model architectures
├── utils/                  # Utility functions
├── data.py                 # Data information script
├── tree_visualizer.py      # Visualization script
├── test.py                 # Testing script
├── train.py                # Training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{ohamouddou2025msdgcnn,
  title={MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification},
  author={Ohamouddou, Said and El Afia, Abdellatif and El Afia, Hanaa and Chiheb, Raddouane},
  journal={arXiv preprint arXiv:2507.12602},
  year={2025}
}
```

## Acknowledgments

This code is based on the excellent implementation of DGCNN: [https://github.com/antao97/dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please open an issue on GitHub or contact the authors.
