import torch
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary

from .depth_decoder_QTR import Depth_Decoder_QueryTr
from .sql_layers import SelfQueryLayer
from ...config.conf import OverfitConf


if __name__=="__main__":

     lt.monkey_patch()
     conf = OverfitConf().conf
     get = lambda x: conf.get(x)

     # DEBUGING EVERY CLASS IN ORDER TO UNDERSTAND ITS FUNCTIONALITY!
     depth_decoder_querytr=True
     full_query_layer=False

     if full_query_layer==True:
          model = SelfQueryLayer()

          x=torch.rand(1, 3, 640, 192)
          K=torch.rand(1, 7, 3)

          y, summary_embedding = model(x, K)
          pytorch_total_params=sum(p.numel() for p in model.parameters() if p.requires_grad)

          print("y.shape: ", y.shape)
          print("summary_embedding.shape: ", summary_embedding.shape)
          print("pytorch_total_params: ", pytorch_total_params) # 0 params!



     elif depth_decoder_querytr==True:
          model = Depth_Decoder_QueryTr(in_channels=get('depth_encoder').get('model_dim'),
                                        patch_size=get('depth_decoder').get('patch_size'),
                                        dim_out=get('depth_decoder').get('dim_out'),
                                        embedding_dim=get('depth_encoder').get('model_dim'),
                                        query_nums=get('depth_decoder').get('query_nums'), num_heads=4,
                                        dim_feedforward=get('depth_decoder').get('dim_feedforward'),
                                        min_val=get('depth_decoder').get('min_depth'), max_val=get('depth_decoder').get('max_depth'))
          #architecture = psummary(model, torch.rand(1, 32, 112, 112), max_depth=4, show_parent_layers=True, print_summary=True)
          #tsummary(model, (32, 112, 112))  # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
          #print(model)
          #y=model(torch.rand(1, 32, 112, 112))
          model.from_pretrained(weights_path='/home/ANT.AMAZON.COM/grigiono/Desktop/SQLdepth/src/pretrained/KITTI_ResNet50_192x640_models/depth.pth', device='cpu')
          y=model(torch.rand(1, 32, 320, 96))

     else:
          raise NotImplementedError(f'Choose only one class or function from the above you want to test or debug!')


