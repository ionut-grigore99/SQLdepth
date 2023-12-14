import torch
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary

from src.models.depth_encoders.convnext.convnextL_encoder import ConvNeXtLEncoderDecoder

if __name__=="__main__":

     lt.monkey_patch()

     # DEBUGING EVERY CLASS IN ORDER TO UNDERSTAND ITS FUNCTIONALITY!
     convnextlargeencoderdecoder=True

     if convnextlargeencoderdecoder==True:
          model = ConvNeXtLEncoderDecoder(pretrained=True, backbone='convnext_large', in_channels=3, num_classes=32, decoder_channels = [1024, 512, 256, 128])
          architecture = psummary(model, torch.rand(1, 3, 1024, 320), max_depth=4, show_parent_layers=True, print_summary=True)
          # architecture = psummary(model, torch.rand(1, 3, 640, 192), max_depth=4, show_parent_layers=True, print_summary=True)
          #architecture = psummary(model, torch.rand(1, 3, 224, 224), max_depth=4, show_parent_layers=True, print_summary=True)
          #tsummary(model, (3, 640, 192))  # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
          # tsummary(model, (3, 224, 224))
          #print(model)
          model.from_pretrained(weights_path='/src/pretrained/KITTI_ConvNeXt_Large_320x1024_models/encoder.pth', device='cpu')
          y=model(torch.rand(1, 3, 1024, 320))
          # y=model(torch.rand(1, 3, 224, 224))
          print(y.shape)

     else:
          raise NotImplementedError(f'Choose only one class or function from the above you want to test or debug!')


