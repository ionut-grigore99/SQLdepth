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

## Inference

```bash
python -m src.inference.test  
```

## Evaluation

```bash
python -m src.evaluation.evaluate_depth
```

### Evaluation of the pretrained models
| Model     | WxH | abs rel | sq rel | RMSE | RMSE log | a1  | a2  | a3  |
|-----------|-----|---------|--------|------|----------|-----|-----|-----|
| KITTI (ResNet-50)    | 640x192 | ...     | ...    |...  | ...      | ... | ... | ... |
| KITTI (ResNet-50) | 1024x320	 | ...     | ...    |...  | ...      | ... | ... | ... |
| KITTI (Efficient-b5) | 1024x320 | ...     | ...    |...  | ...      | ... | ... | ... |
| KITTI (ConvNeXt-L) | 1024x320 | ...     | ...    |...  | ...      | ... | ... | ... |

## Training
```bash
python -m src.train.trainer
```

## Local overfit
```bash
python -m src.overfit.local_trainer
```
