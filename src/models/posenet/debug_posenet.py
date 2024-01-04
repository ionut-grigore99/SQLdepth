'''
FOR DEBUGING AND VERYFING FUNCTIONALITIES OF THE POSE_ENCODER AND POSE_DECODER
'''

import torch
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
# I need to give different names to these 2 summary above

from .pose_cnn import PoseCNN

if __name__=="__main__":

    lt.monkey_patch()

    model=PoseCNN(num_input_frames=2)
    x1=torch.rand(1, 3, 640, 192)
    x2=torch.rand(1, 3, 640, 192)
    input=torch.cat((x1, x2), dim=1)

    ######## 3 WAYS OF VISUALIZING THE ARCHITECTURE ########
    #architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    #tsummary(model, (6, 640, 192)) # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
    # print(model)
    #y=model(input)
    model.from_pretrained(weights_path='/home/ANT.AMAZON.COM/grigiono/Desktop/SQLdepth/src/pretrained/KITTI_EfficientNetB5_320x1024_models/pose.pth', device='cpu')
    y=model(input)
    # breakpoint()

