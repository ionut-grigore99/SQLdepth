import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class OverfitConf:
    name: str = 'overfit_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)


@dataclass
class TrainConf:
    name: str = 'train_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)

##################------yamls for the provided pretrained models------##################
@dataclass
class ResNet50_192x640_Conf:
    name: str = 'resnet50_192x640_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)

@dataclass
class ResNet50_320x1024_Conf:
    name: str = 'resnet50_320x1024_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)

@dataclass
class Effb5_320x1024_Conf:
    name: str = 'effb5_320x104_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)

@dataclass
class ConvNeXtLarge_320x1024_Conf:
    name: str = 'convnextL_320x1024_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)


##################------yamls for the Mamba models------##################

@dataclass
class MambaDepthBot_320x1024_Conf:
    name: str = 'mambaDepthBot_320x1024_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)


@dataclass
class MambaDepthEnc_320x1024_Conf:
    name: str = 'mambaDepthEnc_320x1024_conf.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)