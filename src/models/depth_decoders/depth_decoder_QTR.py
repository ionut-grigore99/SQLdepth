from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from .sql_layers import SelfQueryLayer


class Depth_Decoder_QueryTr(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=16, num_heads=4, query_nums=100, dim_out=256,
                 norm='linear', dim_feedforward=1024, min_val=0.001, max_val=10) -> None:
        super(Depth_Decoder_QueryTr, self).__init__()
        self.norm = norm
        # when we instantiate the SQLDepth, we put in_channels=embedding_dim=model_dim=32!
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True) # Parameter containing:
                                                                                                     #        Parameter[500, 32]

        encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward)
        self.transformer_encoder = nn.modules.transformer.TransformerEncoder(encoder_layers, num_layers=4)
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1) # I don't see exactly the purpose of this

        self.full_query_layer = SelfQueryLayer()
        self.bins_regressor = nn.Sequential(nn.Linear(embedding_dim * query_nums, 16 * query_nums), #this is basically the MLP which generates the bins b.
                                            nn.LeakyReLU(),
                                            nn.Linear(16 * query_nums, 16 * 16),
                                            nn.LeakyReLU(),
                                            nn.Linear(16 * 16, dim_out))

        self.convert_to_prob = nn.Sequential(nn.Conv2d(query_nums, dim_out, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1))
        self.query_nums = query_nums

        self.min_val = min_val
        self.max_val = max_val


    def forward(self, x0):
        # x0 -> tensor[1, 32, 320, 96] (because the model_dim is actually always 32 and here I test for input resolution of (640, 192))
        embeddings_0 = self.embedding_convPxP(x0.clone()) # tensor[1, 32, 20, 6] because 320/16=20 and 96/16=6
        embeddings_0 = embeddings_0.flatten(2) # tensor[1, 32, 120] which is actually (C, H*W/p^2) because 120 = 320*96/256
        embeddings_0 = embeddings_0 + self.positional_encodings[:embeddings_0.shape[2], :].T.unsqueeze(0)
        # embeddings_0 -> tensor[1, 32, 120]
        # self.positional_encodings -> Parameter[500, 32]
        # self.positional_encodings[:embeddings_0.shape[2], :] -> tensor[120, 32]
        # self.positional_encodings[:embeddings_0.shape[2], :].T.unsqueeze(0) -> tensor[1, 32, 120] =>
        # => embeddings_0 -> tensor[1, 32, 120]
        embeddings_0 = embeddings_0.permute(2, 0, 1) # tensor[120, 1, 32]
        total_queries = self.transformer_encoder(embeddings_0) # tensor[120, 1, 32]

        queries = total_queries[:self.query_nums, ...] # tensor[64, 1, 32]
        queries = queries.permute(1, 0, 2) # tensor[1, 64, 32]
                                           # basically each independent query has a dimension of 32 and we have
                                           # a total of query_nums=64 queries.

        x0 = self.conv3x3(x0)  # tensor[1, 32, 320, 96]

        self_cost_volume, summary_embedding = self.full_query_layer(x0, queries)
        # self_cost_volume   -> tensor[1, 64, 320, 96]
        # summary_embedding  -> tensor[1, 64, 32], these represent the "Q depth countings in Q planes".
        #

        # applying MLP to obtain depth bins b.
        bs, Q, E = summary_embedding.shape
        depth_bins = self.bins_regressor(summary_embedding.view(bs, Q * E)) # tensor[1, 64]

        # normalization of the depth bins b, it is not written in the paper about this.
        if self.norm == 'linear': # in our case will go through this if!
            depth_bins = torch.relu(depth_bins) # tensor[1, 64]
            eps = 0.1
            depth_bins = depth_bins + eps # this is done in order to avoid an eventually division by zero after if-else.
        elif self.norm == 'softmax': # I don't know when they use this!
            return torch.softmax(depth_bins, dim=1), self_cost_volume
        else:                        # I don't know when they use this!
            depth_bins = torch.sigmoid(depth_bins)
        depth_bins = depth_bins / depth_bins.sum(dim=1, keepdim=True) # tensor[1, 64]

        # "Firstly, in order to match the dimension of depth bins b of shape D, we apply a 1×1 convolution to the self-cost
        # volume V to obtain a D-planes volume. Secondly, we apply a plane-wise softmax operation to convert the volume into
        # plane-wise probabilistic maps." Basically self.convert_to_prob contains a 1x1 Conv and a Softmax operation.
        plane_wise_probabilistic_maps = self.convert_to_prob(self_cost_volume) # tensor[1, 64, 320, 96]

        bin_widths = (self.max_val - self.min_val) * depth_bins # tensor[1, 64]
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val) # tensor[1, 65]
        bin_edges = torch.cumsum(bin_widths, dim=1) # tensor[1, 65]

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) # tensor[1, 64]
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1) # tensor[1, 64, 1, 1]

        predicted_depth = torch.sum(plane_wise_probabilistic_maps * centers, dim=1, keepdim=True) # tensor[1, 1, 320, 96]

        return predicted_depth # tensor[1, 1, 320, 96]

    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict_dec = torch.load(weights_path, map_location=device)
        filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict_dec)
        self.eval()
