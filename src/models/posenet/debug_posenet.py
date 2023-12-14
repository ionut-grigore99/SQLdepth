'''
FOR DEBUGING AND VERYFING FUNCTIONALITIES OF THE POSE_ENCODER AND POSE_DECODER
'''

import torch
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
# I need to give different names to these 2 summary above

from .pose_cnn import PoseCNN
from .pose_decoder import PoseDecoder

if __name__=="__main__":

    lt.monkey_patch()

    pose_cnn=True

    if pose_cnn==True:
        model=PoseCNN(num_input_frames=2)
        x1=torch.rand(1, 3, 640, 192)
        x2=torch.rand(1, 3, 640, 192)
        input=torch.cat((x1, x2), dim=1)

        ######## 3 WAYS OF VISUALIZING THE ARCHITECTURE ########
        #architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
        tsummary(model, (6, 640, 192)) # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
        # print(model)
        #y=model(input)

    else:
        model=PoseDecoder(num_ch_enc=69, num_input_features=69) #AICI MAI TREBUIE SA VAD CE SI CUM
        x1=torch.rand(1, 3, 640, 192)
        x2=torch.rand(1, 3, 640, 192)
        input=torch.cat((x1, x2), dim=1)

        ######## 3 WAYS OF VISUALIZING THE ARCHITECTURE ########
        # architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
        tsummary(model, (6, 640, 192))
        # print(model)
        #y = model(input)
