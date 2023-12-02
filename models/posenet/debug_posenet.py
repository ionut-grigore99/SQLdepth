'''
FOR DEBUGING AND VERYFING FUNCTIONALITIES OF THE POSE_ENCODER AND POSE_DECODER
'''

import torch
import lovely_tensors as lt
from pytorch_model_summary import summary
from torchsummary import summary

from .pose_cnn import PoseCNN
from .pose_decoder import PoseDecoder

if __name__=="__main__":

    lt.monkey_patch()

    pose_cnn=False

    if pose_cnn==True:
        model=PoseCNN(num_input_frames=2)
        x1=torch.rand(1, 3, 640, 192)
        x2=torch.rand(1, 3, 640, 192)
        input=torch.cat((x1, x2), dim=1)

        ######## 3 WAYS OF VISUALIZING THE ARCHITECTURE ########
        #architecture = summary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
        summary(model, (6, 640, 192)) # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
        # print(model)
        #y=model(input)

    else:
        model=PoseDecoder(num_ch_enc=69, num_input_features=69) #AICI MAI TREBUIE SA VAD CE SI CUM
        x1=torch.rand(1, 3, 640, 192)
        x2=torch.rand(1, 3, 640, 192)
        input=torch.cat((x1, x2), dim=1)

        ######## 3 WAYS OF VISUALIZING THE ARCHITECTURE ########
        # architecture = summary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
        summary(model, (6, 640, 192))
        # print(model)
        #y = model(input)
