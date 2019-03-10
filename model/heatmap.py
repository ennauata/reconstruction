from model.drn import drn_d_54
from model.modules import PyramidModule, ConvBlock
import torch

class HeatmapModel(torch.nn.Module):
    def __init__(self, options, num_classes=1):
        super(HeatmapModel, self).__init__()
        
        self.options = options        
        self.drn = drn_d_54(pretrained=True, out_map=32, num_classes=-1, out_middle=False)
        self.pyramid = PyramidModule(options, 512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.heatmap_pred = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(256, 256), mode='bilinear')
        return

    def forward(self, inp):
        features = self.drn(inp)
        features = self.pyramid(features)
        features = self.feature_conv(features)
        heatmap_pred = self.heatmap_pred(features)
        heatmap_pred = self.upsample(heatmap_pred)
        heatmap_pred = torch.sigmoid(heatmap_pred).squeeze(1)
        return heatmap_pred
