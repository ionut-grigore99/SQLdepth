import torch
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
from ..config.conf import OverfitConf, ResNet50_192x640_Conf, ResNet50_320x1024_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf

from .SQLDepth import SQLdepth


if __name__=="__main__":

     lt.monkey_patch()

     conf = ResNet50_320x1024_Conf().conf

     model = SQLdepth(conf)
     #architecture = psummary(model, torch.rand(1, 3, 640, 192), max_depth=4, show_parent_layers=True, print_summary=True)
     #architecture = psummary(model, torch.rand(1, 3, 224, 224), max_depth=4, show_parent_layers=True, print_summary=True)
     #tsummary(model, (3, 640, 192))  # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
     #tsummary(model, (3, 224, 224))
     #print(model)
     #y=model(torch.rand(1, 3, 640, 192))
     #y=model(torch.rand(1, 3, 224, 224))
     model.from_pretrained()
     # y=model(torch.rand(1, 3, 224, 224))

     if conf == ResNet50_192x640_Conf:
          y = model(torch.rand(1, 3, 640, 192))
     else:
          y = model(torch.rand(1, 3, 1024, 320))
     breakpoint()




