import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
import math
from einops import rearrange, repeat


class LinearizedSematicGraphConvolutional(nn.Module):
    def __init__(self, hidden_dim, order=1):
        super(LinearizedSematicGraphConvolutional, self).__init__()
        self.w = nn.Parameter(torch.empty((hidden_dim, (order + 1) * hidden_dim)))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=0.1)
        self.order = order

    def linearized_gconv(self, x, nodevec):
        # AH
        hidden = torch.einsum("nc,bnh->bhc", nodevec, x)
        hidden = torch.einsum("nc,bhc->bnh", nodevec, hidden)

        # D
        degree = torch.ones([nodevec.shape[0]]).to(nodevec.device)
        degree = torch.einsum("nc,n->c", nodevec, degree)
        degree = torch.einsum("nc,c->n", nodevec, degree)
        degree = degree.unsqueeze(-1) + 1e-6

        # D^(-1)AH = AH/D
        hidden = hidden / degree

        return hidden.contiguous()

    def forward(self, x, nodevec): 
        # x: [B, N, C]
        out = [x]

        x1 = self.linearized_gconv(x, nodevec)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.linearized_gconv(x1, nodevec)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=-1)
        h = torch.matmul(h, self.w.T)
        h = self.dropout(h)

        return h


class SSE(nn.Module):
    def __init__(self, p=0.2):
        """
        Args:
            p (float): Replacement probability, value range [0, 1]. The larger p is, the more frequent the replacement is.
        """
        super(SSE, self).__init__()
        self.p = p

    def forward(self, embeddings):
        """
        Args:
            embeddings (Tensor): embedding matrix of shape [N, C]
        Returns:
            Tensor: The embedding matrix after applying SSE regularization.
        """
        # If in evaluation mode or p is 0, the original embeddings are returned directly.
        if not self.training or self.p == 0:
            return embeddings

        N, C = embeddings.shape

        # Generate a Bernoulli mask for each node to decide whether to replace it
        mask = torch.rand(N, device=embeddings.device) < self.p

        # Generate a random index for each node to replace
        random_indices = torch.randint(0, N, (N,), device=embeddings.device)

        # If mask is True, replace it with a randomly selected embedding, otherwise keep the original value
        mask = mask.unsqueeze(1)  # [N, 1]
        sse_embeddings = torch.where(mask, embeddings[random_indices], embeddings)

        return sse_embeddings


class GatedChannelMLP(nn.Module):
    """
    Gated Channel MLP

    Dynamic channel-wise feature transformation.

    Input:
        x: [B,N,D]

    Output:
        x: [B,N,D]

    """

    def __init__(
        self,
        dim,
        expansion=4,
        dropout=0.1
    ):
        super().__init__()


        hidden_dim = int(dim * expansion)


        # ============================
        # Value branch
        #
        # feature transformation
        #
        # D -> 4D
        # ============================

        self.value_proj = nn.Linear(
            dim,
            hidden_dim
        )


        # ============================
        # Gate branch
        #
        # learn feature importance
        #
        # D -> 4D
        # ============================

        self.gate_proj = nn.Linear(
            dim,
            hidden_dim
        )


        # ============================
        # Output projection
        #
        # 4D -> D
        # ============================

        self.output_proj = nn.Linear(
            hidden_dim,
            dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.norm = nn.LayerNorm(
            dim
        )



    def forward(self,x):

        """
        x:
        [B,N,D]
        """


        residual = x


        # pre-normalization

        x = self.norm(x)


        # ============================
        # gated feature transformation
        # ============================

        value = self.value_proj(x)


        gate = torch.sigmoid(
            self.gate_proj(x)
        )


        # adaptive channel selection

        x = value * gate


        # non-linear transformation

        x = F.gelu(x)


        # project back

        x = self.output_proj(x)


        x = self.dropout(x)


        # residual connection

        return residual + x


class PhysicalGraphConvolution(nn.Module):

    def __init__(
        self,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()


        self.proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.norm = nn.LayerNorm(
            hidden_dim
        )


    def forward(
        self,
        x,
        adj
    ):

        """
        x:
            [B,N,D]

        adj:
            [N,N]
            normalized physical adjacency
        """


        # AX

        h = torch.einsum(
            "nm,bmd->bnd",
            adj,
            x
        )


        h = self.proj(h)

        h = F.gelu(h)

        h = self.dropout(h)

        h = self.norm(h)


        return h

class TemporalMixer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        hidden_dim = feature_dim * 2
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=feature_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, input_data):
        # input_data: [B, N, C]
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data

        return hidden

class S_MLPMixerBlock(nn.Module):
    def __init__(self, hidden_dim, rank, order=1, top_k_moe=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.cosGCN = LinearizedSematicGraphConvolutional(hidden_dim, order)
        self.phyGCN = PhysicalGraphConvolution(hidden_dim, dropout=0.1)
        self.gama = nn.Parameter(torch.tensor(0.1))
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mixer = GatedChannelMLP(dim=hidden_dim, expansion=4, dropout=0.1)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.temporal_mixer = TemporalMixer(self.hidden_dim)

    def forward(self, x, nodevec, adj):
        # node -> group
        x = self.temporal_mixer(x)
        hidden_sem = self.cosGCN(x, nodevec)  # [B,N,D]
        hidden_phy = self.phyGCN(x, adj)
        hidden_graph = self.gama * hidden_phy + (1 - self.gama) * hidden_sem
        x = self.beta * x + (1 - self.beta) * hidden_graph
        # channel mixing
        y_c = self.channel_mixer(x)
        x = x + self.alpha * y_c
        return x


class PSGCN(nn.Module):
    def __init__(self, args, num_nodes, adj_mx):
        super().__init__()

        # attributes
        self.num_nodes = num_nodes
        self.spatial_dim = args.spatial_dim
        self.input_len = args.seq_len
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.embed_dim = args.embed_dim
        self.pred_len = args.pred_len
        self.num_layer = args.num_layer
        self.temp_dim_tid = args.temp_dim_tid
        self.temp_dim_diw = args.temp_dim_diw
        self.time_of_day_size = args.steps_per_day
        self.day_of_week_size = args.day_of_week_size
        self.rank = args.rank
        self.adj = adj_mx
        self.hidden_dim = self.spatial_dim + self.temp_dim_tid + self.temp_dim_diw + self.embed_dim

        # ablation attributes
        self.cos_gconv = args.cos_gconv
        self.softmax_gconv = args.softmax_gconv
        self.geo_gconv = args.geo_gconv

        self.time_series_emb_layer = nn.Linear(self.input_dim * self.input_len, self.embed_dim)

        # spatial embeddings
        self.node_emb = nn.Parameter(
            torch.empty(self.num_nodes, self.spatial_dim))  # N,d
        nn.init.xavier_uniform_(self.node_emb)
        if args.sse:
            self.sse = SSE(p=args.p) 
        elif args.dropout:
            self.sse = nn.Dropout(p=args.p)
        else:
            self.sse = nn.Identity()

        # temporal embeddings
        self.time_in_day_emb = nn.Parameter(
            torch.empty(self.time_of_day_size, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
            torch.empty(self.day_of_week_size, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        if self.cos_gconv:
            self.emb_w1 = nn.Parameter(torch.randn(self.spatial_dim, self.spatial_dim))
            self.emb_w2 = nn.Parameter(torch.randn(self.spatial_dim, self.spatial_dim))

        self.encoder = nn.ModuleList([
            S_MLPMixerBlock(self.hidden_dim, self.spatial_dim, order=1, top_k_moe=3)
            for _ in range(self.num_layer)
        ])
        self.regression_layer = nn.Linear(self.hidden_dim, self.pred_len)

    def compute_adj(self, nodevec):
        adj = torch.mm(nodevec, nodevec.T)
        rowsum = torch.sum(adj, dim=1) + 1e-6

        return adj / rowsum

    def laplacian_smoothing(self):
        adj = self.adj
        X = self.node_emb
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L.float()), X)
        loss_smooth_feat = torch.trace(XLXT)

        return loss_smooth_feat

    def forward(self, history_data):
        input_data = history_data[..., range(self.input_dim)]
        temp_data = input_data

        x_last = input_data[:, -1:, :, :1].repeat([1, self.input_len, 1, 1])
        batch_size, _, num_nodes, _ = input_data.shape
        # time series embedding
        input_data = input_data.transpose(1, 2).contiguous()  # B,N,T,D
        input_data = input_data.view(
            batch_size, num_nodes, -1)  # B,N,TD
        # time in day
        t_i_d_data = history_data[..., -2]  # B,T,N
        # day of week
        d_i_w_data = history_data[..., -1]  # B,T,N

        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        stochastic_shared_node_emb = self.sse(self.node_emb)  
        node_emb.append(stochastic_shared_node_emb.unsqueeze(0).expand(batch_size, -1, -1))  # B,N,d

        # vis
        adj_list = []
        if self.cos_gconv:  
            gate = F.softmax(F.relu(torch.matmul(stochastic_shared_node_emb, self.emb_w1)), dim=1)
            filter = F.relu(torch.matmul(stochastic_shared_node_emb, self.emb_w2))
            support = F.normalize(gate * filter, p=2, dim=1) 
            adj_list.append(self.compute_adj(support))
        elif self.softmax_gconv:
            support = F.softmax(F.relu(torch.mm(stochastic_shared_node_emb, stochastic_shared_node_emb.T)), dim=1)
            adj_list.append(support.detach())
        elif self.geo_gconv:
            support = self.adj

        tem_emb_node = []
        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, 0:1] * self.time_of_day_size).type(torch.LongTensor)]
        tem_emb_node.append(time_in_day_emb.expand(batch_size, self.num_nodes, -1))  # B,N,d
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, 0:1] * self.day_of_week_size).type(torch.LongTensor)]
        tem_emb_node.append(day_in_week_emb.expand(batch_size, self.num_nodes, -1))  # B,N,d

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb_node, dim=2)

        x = hidden
        for block in self.encoder:
            x = block(x, support, self.adj)

        pred = self.regression_layer(x).transpose(1, 2).contiguous().unsqueeze(-1) + x_last
        if self.training:
            return pred, adj_list
        else:
            return pred