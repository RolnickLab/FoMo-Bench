import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_cluster import knn_graph
from torch_geometric.nn import PPFConv
from torch_cluster import fps


class PointNetLayer(MessagePassing):
    def __init__(self, in_dim, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + in_dim, out_channels), ReLU(), Linear(out_channels, out_channels))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.


class PointNet(torch.nn.Module):
    def __init__(self, nb_classes, in_channels=3, in_dim=3):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(in_dim, in_channels, 32)
        self.conv2 = PointNetLayer(in_dim, 32, 32)
        self.classifier = Linear(32, nb_classes)

    def forward(self, pos, batch=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        # h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


class PPFNet(torch.nn.Module):
    def __init__(self, nb_classes, in_channels=3, n_dims=3):
        super().__init__()

        # torch.manual_seed(12345)
        mlp1 = Sequential(Linear(in_channels + n_dims, 32), ReLU(), Linear(32, 32))
        self.conv1 = PPFConv(mlp1)  # TODO
        mlp2 = Sequential(Linear(32 + n_dims, 32), ReLU(), Linear(32, 32))
        self.conv2 = PPFConv(mlp2)
        self.classifier = Linear(32, nb_classes)

    def forward(self, pos, normal, batch=None):
        edge_index = knn_graph(pos, k=16, batch=batch, loop=False)

        x = self.conv1(x=None, pos=pos, normal=normal, edge_index=edge_index)
        x = x.relu()
        x = self.conv2(x=x, pos=pos, normal=normal, edge_index=edge_index)
        x = x.relu()

        x = global_max_pool(x, batch)  # [num_examples, hidden_channels]
        return self.classifier(x)
