import torch
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary

from src.models.depth_encoders.resnet.resnet_encoder import resnet_multiimage_input, ResnetEncoder, ResnetEncoderDecoder

if __name__=="__main__":

     lt.monkey_patch()

     # DEBUGING EVERY CLASS IN ORDER TO UNDERSTAND ITS FUNCTIONALITY!
     resnetmultiimageinput=False
     resnetencoder=False
     resnetencoderdecoder=True

     if resnetmultiimageinput==True:
          #model = resnet_multiimage_input(num_layers=18, pretrained=False, num_input_images=1)
          model = resnet_multiimage_input(num_layers=50, pretrained=False, num_input_images=1)
          #architecture = psummary(model, torch.rand(1, 3, 640, 192), max_depth=4, show_parent_layers=True, print_summary=True)
          #architecture = psummary(model, torch.rand(1, 3, 640, 192), max_depth=4, show_parent_layers=True, print_summary=True)
          #tsummary(model, (3, 640, 192))  # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
          tsummary(model, (3, 224, 224))
          #print(model)

     elif resnetencoder==True:
          #model=ResnetEncoder(num_layers=18, pretrained=False, num_input_images=1)
          model = ResnetEncoder(num_layers=50, pretrained=False, num_input_images=1)
          #architecture = psummary(model, torch.rand(1, 3, 640, 192), max_depth=4, show_parent_layers=True, print_summary=True)
          #architecture = psummary(model, torch.rand(1, 3, 224, 224), max_depth=4, show_parent_layers=True, print_summary=True)
          #tsummary(model, (3, 640, 192))  # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
          # tsummary(model, (3, 224, 224))
          #print(model)
          #y=model(torch.rand(1, 3, 640, 192))
          y=model(torch.rand(1, 3, 224, 224))

     elif resnetencoderdecoder==True:
          #model=Resnet18EncoderDecoder(num_layers=18, num_features=256, model_dim=32)
          model = ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)
          #architecture = psummary(model, torch.rand(1, 3, 640, 192), max_depth=4, show_parent_layers=True, print_summary=True)
          #architecture = psummary(model, torch.rand(1, 3, 224, 224), max_depth=3, show_parent_layers=True, print_summary=True)
          # tsummary(model, (3, 640, 192))  # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
          # tsummary(model, (3, 224, 224)) # this not work properly for encoderdecoder class here
          # tisummary(model, input_size=(1, 3, 224, 224)) # this is a bit weird tbh
          # print(model)
          #y=model(torch.rand(1, 3, 640, 192))
          #y=model(torch.rand(1, 3, 224, 224))
          model.from_pretrained(weights_path='/home/ANT.AMAZON.COM/grigiono/Desktop/SQLdepth/src/pretrained/KITTI_ResNet50_192x640_models/encoder.pth', device='cpu')
          # y=model(torch.rand(1, 3, 224, 224))
          y = model(torch.rand(1, 3, 192, 640))
          print(y.shape)

     else:
          raise NotImplementedError(f'Choose only one class or function from the above you want to test or debug!')


