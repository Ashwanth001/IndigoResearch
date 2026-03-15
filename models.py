import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Dropout, GRU
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData

class HomogeneousGNN(torch.nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class BipartiteEncoder(torch.nn.Module):
    def __init__(self, country_in_dim, product_in_dim, hidden_dim=128, metadata=None):
        super().__init__()
        self.country_lin = Linear(country_in_dim, hidden_dim)
        self.product_lin = Linear(product_in_dim, hidden_dim)
        
        # The base GNN expects hidden_dim-sized features for all nodes
        self.gnn = to_hetero(HomogeneousGNN(hidden_dim), metadata)

    def forward(self, x_dict, edge_index_dict):
        # Initial projection to common hidden dimension
        x_dict = {
            'country': self.country_lin(x_dict['country']),
            'product': self.product_lin(x_dict['product'])
        }
        # Apply Heterogeneous GNN
        return self.gnn(x_dict, edge_index_dict)

class TemporalBipartiteGNN(torch.nn.Module):
    def __init__(self, encoder: BipartiteEncoder, hidden_dim=128, temporal_hidden_dim=128, num_layers_gru=1):
        super().__init__()
        self.encoder = encoder
        self.temporal_hidden_dim = temporal_hidden_dim
        
        self.gru_country = GRU(hidden_dim, temporal_hidden_dim, num_layers=num_layers_gru)
        self.gru_product = GRU(hidden_dim, temporal_hidden_dim, num_layers=num_layers_gru)

    def forward(self, snapshots: list[HeteroData]) -> dict[str, Tensor]:
        country_seq = []
        product_seq = []
        
        # 1. Map each snapshot to embeddings
        for snap in snapshots:
            z_dict = self.encoder(snap.x_dict, snap.edge_index_dict)
            country_seq.append(z_dict['country'])
            product_seq.append(z_dict['product'])
            
        # 2. Sequence across years: [T, N, hidden_dim]
        country_seq = torch.stack(country_seq, dim=0)
        product_seq = torch.stack(product_seq, dim=0)
        
        # 3. GRU independently for countries and products
        # GRU input: [seq_len, batch, input_size]. Here batch = number of nodes.
        z_country_all, _ = self.gru_country(country_seq)
        z_product_all, _ = self.gru_product(product_seq)
        
        # 4. Final snapshot temporal embedding: [N, temporal_hidden_dim]
        # Taking last time point T=5
        return {
            'country': z_country_all[-1],
            'product': z_product_all[-1]
        }

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_dim, 1)
        )

    def forward(self, z_country, z_product, edge_index):
        # edge_index is [2, E] where row 0 is country idx and row 1 is product idx
        c_idx, p_idx = edge_index[0], edge_index[1]
        
        z_c = z_country[c_idx]
        z_p = z_product[p_idx]
        
        # Concatenate features: [E, in_dim]
        x = torch.cat([z_c, z_p], dim=-1)
        return self.mlp(x).view(-1)
