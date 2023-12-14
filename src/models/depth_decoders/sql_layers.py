import torch
import torch.nn as nn


class SelfQueryLayer(nn.Module):
    def __init__(self) -> None:
        super(SelfQueryLayer, self).__init__()

    def forward(self, x, queries):
        """
        Given feature map of size [bs, E, H, W] (feature map S), and queries of size [bs, Q, E] (coarsed-grained queries where E
        is embedding_dim which is actually chosen as model_dim=32 and Q is query_nums=64), return Q self-cost_volume planes corresponding
        to Q queries (forming the self-cost_volume of shape [bs, Q, H, W], where each of the Q self-cost_volume plane has the shape [bs, 1, H, W])
        and summary_embedding of shape [bs, Q, E].
        """
        bs, E, h, w = x.size()  # bs, E, H, W
        _, Q, E_Q = queries.size()  # bs, Q, E
        assert E == E_Q, "Number of channels in x and embedding dimension (at dim 2) of queries matrix must match"


        y = torch.matmul(x.view(bs, E, h * w).permute(0, 2, 1), queries.permute(0, 2, 1))  # (bs, H*W, Q)
        y_norm = torch.softmax(y, dim=1) # (bs, H*W, Q)
        y = y.permute(0, 2, 1).view(bs, Q, h, w)  # (bs, Q, H, W)

        summary_embedding = torch.matmul(y_norm.permute(0, 2, 1), x.view(bs, E, h * w).permute(0, 2, 1)) # (bs, Q, E)
        # these summary_embeddings are the input which goes to MLP in order to generate the bins b.
        # they represent the "Q depth countings in Q planes".

        return y, summary_embedding # [(bs, Q, H, W), (bs, Q, E)]
