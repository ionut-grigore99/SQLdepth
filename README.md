# SQLdepth


## Introduction
This is the implementation of the paper called "*SQLdepth: Generalizable Self-Supervised Fine-Structured Monocular Depth
Estimation*" (https://arxiv.org/pdf/2309.00526.pdf) for my disertation thesis.

## Instalation

First create a conda environment called **sqldepth**:
```bash
conda create --name sqldepth --clone base 
```
or without cloning the base:
```bash
conda create --name sqldepth
```

Activate the new enviroment:
```bash
conda activate sqldepth
```

After that, run the following:
```bash
pip install -e .
```
or
```bash
pip install -e . && pip install -e ".[dev]"
```

Recommended to install the [dev] dependencies.

## KITTI training data
You can download the entire [raw KITTI dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```bash
wget -i src/data/kitti/kitti_archives_to_download.txt -P src/data/kitti/kitti_data/
```
The **'-i'** option in wget stands for "input-file". <br />
This option specifies a file that contains a list of URLs to download.  <br />
In this case, the file is src/data/kitti/kitti_archives_to_download.txt.  <br />
This file should contain a list of URLs, each on a new line, pointing to the files that need to be downloaded. <br />

<br />

The **'-P'** option specifies the prefix (directory) where downloaded files will be saved. <br /> 
In this case, the files are being downloaded to the src/data/kitti/kitti_data/ directory. <br />

<br />
Then unzip with:

```bash
cd src/data/kitti/kitti_data
unzip "*.zip"
cd .. # 4 times
```
<br />

**Warning**: it weighs about 175GB, so make sure you have enough space to unzip too!
<br />

Their default settings expect that you have converted the png images to jpeg with this command, **which also deletes
the raw KITTI `.png` files**:
```bash
find src/data/kitti/kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.
## Inference

```bash
python -m src.inference.test  
```

## Evaluation

```bash
python -m src.evaluation.evaluate_depth
```

### Evaluation of the pretrained models

#### Evaluation split: eigen
| Model     | Params | WxH | abs rel | sq rel | RMSE  | RMSE log | a1    | a2    | a3    |
|-----------|--------|-----|---------|--------|-------|-------|-------|-------|-------|
| KITTI (ResNet-50)    | 31,051,944  | 640x192 | 0.088   | 0.698  | 4.175 | 0.167 | 0.919 | 0.969 | 0.984 |
| KITTI (ResNet-50) | 37,782,824  | 1024x320	 | 0.082   | 0.607  | 3.914 | 0.160 | 0.928 | 0.972 | 0.985 |
| KITTI (Efficient-b5) | 45,702,416  | 1024x320 | 0.084   | 0.539  | 3.897 | 0.162 | 0.924 | 0.971 | 0.985 |
| KITTI (ConvNeXt-L) | 242,150,304  | 1024x320 | 0.074   | 0.491  | 3.578 | 0.150 | 0.939 | 0.974 | 0.986 |

#### Paper reported results:
![img.png](assets/paper_reported_results_eigen.png)
#### Evaluation split: eigen_benchmark
| Model     | Params | WxH | abs rel | sq rel | RMSE  | RMSE log | a1    | a2    | a3    |
|-----------|--------|-----|-------|-------|-------|-------|-------|-------|-------|
| KITTI (ResNet-50)    | 31,051,944  | 640x192 | 0.054 | 0.276 | 2.819 | 0.092 | 0.964 | 0.993 | 0.998 |
| KITTI (ResNet-50) | 37,782,824  | 1024x320	 | 0.052 | 0.223 | 2.550 | 0.084 | 0.971 | 0.995 | 0.998 |
| KITTI (Efficient-b5) | 45,702,416  | 1024x320 | 0.058 | 0.243 | 2.825 | 0.093 | 0.964 | 0.994 | 0.998 |
| KITTI (ConvNeXt-L) | 242,150,304  | 1024x320 | 0.045 | 0.147 | 2.130 | 0.070 | 0.982 | 0.997 | 0.999 |

#### Paper reported results:
![img_1.png](assets/paper_reported_results_eigen_benchmark.png)
## Training
```bash
python -m src.train.trainer
```

## Local overfit

The goal was to see if the SQLDepth architecture could be
overfit on a small batch of samples—typically one, two, or five images.
The ability to overfit is a litmus test for model capacity, and visual 
aids, such as the rendering of predicted depth maps, were instrumental in evaluating 
the success. If overfitting was achieved with satisfactory visual 
confirmation, the next logical step involved deploying the entire 
training pipeline on my ec2 instance, utilizing the full dataset.
```bash
python -m src.overfit.local_trainer
```
