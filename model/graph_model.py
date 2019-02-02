import torch
import torch.nn as nn
from torch_geometric.nn import SplineConv, GCNConv

class GraphModel(torch.nn.Module):
    def __init__(self, options):
        super(GraphModel, self).__init__()
        #self.corner_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        #self.connection_encoder = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.corner_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.connection_encoder = nn.Sequential(nn.Linear(5, 32), nn.ReLU())        

        if options.conv_type == 'gcn':
            self.conv_1 = GCNConv(32, 64)
            self.conv_2 = GCNConv(64, 64)
            self.conv_3 = GCNConv(64, 64)
        else:
            self.conv_1 = SplineConv(32, 64, dim=1, kernel_size=2)
            self.conv_2 = SplineConv(64, 64, dim=1, kernel_size=2)
            self.conv_3 = SplineConv(64, 64, dim=1, kernel_size=2)
            pass

        self.corner_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.connection_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        return
    
    def forward(self, corners, connections, edge_index, edge_attr):
        corner_x = self.corner_encoder(torch.cat([corners, torch.zeros(len(corners), 1).cuda()], dim=-1))
        #connection_x = self.connection_encoder(torch.cat([connections, torch.ones(len(connections), 1).cuda()], dim=-1))
        connection_x = self.connection_encoder(connections)
        x = torch.cat([corner_x, connection_x], 0)
        x = self.conv_1(x, edge_index, edge_attr)
        x = self.conv_2(x, edge_index, edge_attr)
        x = self.conv_3(x, edge_index, edge_attr)                
        
        return torch.sigmoid(self.corner_pred(x[:len(corners)])).view(-1), torch.sigmoid(self.connection_pred(x[len(corners):])).view(-1)
