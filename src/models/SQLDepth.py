import torch
import torch.nn as nn
import os

from src.models.depth_encoders.resnet.resnet_encoder import ResnetEncoderDecoder
from src.models.depth_encoders.efficientnet.efficientnet_encoder import EfficientNetEncoderDecoder
from src.models.depth_encoders.convnext.convnextL_encoder import ConvNeXtLEncoderDecoder
from .depth_decoders.depth_decoder_QTR import Depth_Decoder_QueryTr


class SQLdepth(nn.Module):
    def __init__(self, conf):
        super(SQLdepth, self).__init__()
        self.conf = conf
        self.get = lambda x: conf.get(x)

        if self.get('depth_encoder').get('model_type') == "resnet":
            self.encoder = ResnetEncoderDecoder(num_layers=self.get('depth_encoder').get('num_layers'),
                                                num_features=self.get('depth_encoder').get('num_features'),
                                                model_dim=self.get('depth_encoder').get('model_dim'))
        elif self.get('depth_encoder').get('model_type') == "efficientnet":
            self.encoder = EfficientNetEncoderDecoder(pretrained=True, backbone='tf_efficientnet_b5_ap',
                                                      in_channels=3,
                                                      num_classes=self.get('depth_encoder').get('model_dim'),
                                                      decoder_channels=self.get('depth_encoder').get('dec_channels'))
        elif self.get('depth_encoder').get('model_type') == "convnextL":
            self.encoder = ConvNeXtLEncoderDecoder(pretrained=True, backbone='convnext_large',
                                                   in_channels=3,
                                                   num_classes=self.get('depth_encoder').get('model_dim'),
                                                   decoder_channels=self.get('depth_encoder').get('dec_channels'))
        else:
            raise NotImplementedError(f'Choose from resnet, efficientnet or convnextL possibilities!')

        self.depth_decoder = Depth_Decoder_QueryTr(in_channels=self.get('depth_encoder').get('model_dim'),
                                                   patch_size=self.get('depth_decoder').get('patch_size'),
                                                   dim_out=self.get('depth_decoder').get('dim_out'),
                                                   embedding_dim=self.get('depth_encoder').get('model_dim'),
                                                   query_nums=self.get('depth_decoder').get('query_nums'), num_heads=4,
                                                   dim_feedforward=self.get('depth_decoder').get('dim_feedforward'),
                                                   min_val=self.get('depth_decoder').get('min_depth'), max_val=self.get('depth_decoder').get('max_depth'))

    def forward(self, x):
        x = self.encoder(x)
        return self.depth_decoder(x)

    def from_pretrained(self):
        print("-> Loading pretrained encoder from ", self.get('pretrained_models_folder'))
        self.encoder.from_pretrained(weights_path=os.path.join(self.get('pretrained_models_folder'), "encoder.pth"),
                                     device= torch.device("cuda" if self.get('use_cuda') else "cpu"))

        print("-> Loading pretrained depth decoder from ", self.get('pretrained_models_folder'))
        self.depth_decoder.from_pretrained(weights_path=os.path.join(self.get('pretrained_models_folder'), "depth.pth"),
                                     device= torch.device("cuda" if self.get('use_cuda') else "cpu"))
