import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if out.shape == residual.shape:
            out += residual
            pass
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def CustomResNet(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet34'])
        state = model.state_dict()
        new_state_dict = state
        for k, v in state_dict.items():
            if k in state and v.shape == state[k].shape:
                new_state_dict[k] = v
                continue
            if len(v.shape) == 4:
                if state[k].shape[1] < v.shape[1]:
                    new_state_dict[k] = v[:, :state[k].shape[1]]
                else:
                    new_state_dict[k] = torch.cat([v, v[:, :1].repeat((1, state[k].shape[1] - v.shape[1], 1, 1))], dim=1)
                    pass
                pass
            continue
        state.update(new_state_dict)
        model.load_state_dict(state)
        pass
    return model

class ResNetBatch(nn.Module):
    def __init__(self, options, block, layers, num_classes=1):
        super(ResNetBatch, self).__init__()
        self.options = options
        self.inplanes = 64

        self.relu = nn.ReLU(inplace=True)            
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
            
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            if _ == blocks - 1 and 'sharing' in self.options.suffix:
                layers.append(block(self.inplanes, planes // 4))
            else:
                layers.append(block(self.inplanes, planes))
                pass
            continue

        return nn.Sequential(*layers)

    def aggregate(self, x, left_edges, right_edges):
        left_x = torch.zeros(x.shape).cuda()
        left_x.index_add_(0, left_edges[:, 0], x[left_edges[:, 1]])
        right_x = torch.zeros(x.shape).cuda()
        right_x.index_add_(0, right_edges[:, 0], x[right_edges[:, 1]])
        global_x = torch.zeros(x.shape).cuda()
        global_x[:] = x.sum(0)
        x = torch.cat([x, left_x, right_x, global_x], dim=1)
        return x
        
    def forward(self, x, left_edges, right_edges):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if 'sharing' in self.options.suffix:
            x = self.layer1(x)
            x = self.aggregate(x, left_edges, right_edges)
            x = self.layer2(x)
            x = self.aggregate(x, left_edges, right_edges)            
            x = self.layer3(x)
            x = self.aggregate(x, left_edges, right_edges)            
            x = self.layer4(x)
            x = self.aggregate(x, left_edges, right_edges)            
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            pass

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return torch.sigmoid(x).view(-1)

def create_model(options):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetBatch(options, BasicBlock, [3, 4, 6, 3], num_classes=1)
    if options.restore == 0:
        state_dict = model_zoo.load_url(model_urls['resnet34'])
        state = model.state_dict()
        new_state_dict = state
        for k, v in state_dict.items():
            if k in state and v.shape == state[k].shape:
                new_state_dict[k] = v
                continue
            if len(v.shape) == 4:
                if state[k].shape[1] < v.shape[1]:
                    v = v[:, :state[k].shape[1]]
                elif state[k].shape[1] > v.shape[1]:
                    v = torch.cat([v, v[:, :1].repeat((1, state[k].shape[1] - v.shape[1], 1, 1))], dim=1)
                    pass
                if state[k].shape[0] < v.shape[0]:
                    new_state_dict[k] = v[:state[k].shape[0]]
                elif state[k].shape[0] > v.shape[0]:
                    new_state_dict[k] = torch.cat([v, v[:1].repeat((state[k].shape[0] - v.shape[0], 1, 1, 1))], dim=0)
                else:
                    new_state_dict[k] = v
                    pass
                pass
            continue
        model.load_state_dict(state)
        pass
    return model

class GraphModelCustom(nn.Module):
    def __init__(self, options):
        super(GraphModelCustom, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.layer_3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.layer_4 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())

        self.inplanes = 16
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.image_layer_0 = nn.Sequential(nn.Conv2d(4, 16, kernel_size=7, stride=2, padding=3,
                                                     bias=False),
                                           nn.BatchNorm2d(16),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.image_layer_1 = self._make_layer(block, 16, layers[0])
        self.image_layer_2 = self._make_layer(block, 32, layers[1], stride=2)
        self.image_layer_3 = self._make_layer(block, 64, layers[2], stride=2)
        self.image_layer_4 = self._make_layer(block, 128, layers[3], stride=2)        
        self.connection_pred = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))

        self.num_edge_points = 8
        return

    def image_features(self, image_x, connections):
        width = image_x.shape[3]
        height = image_x.shape[3]
        x_1 = connections[:, 1:2] * width
        x_2 = connections[:, 3:4] * width
        y_1 = connections[:, 0:1] * height
        y_2 = connections[:, 2:3] * height
        alphas = (torch.arange(self.num_edge_points).float() / (self.num_edge_points - 1)).cuda()
        xs = torch.round(x_1 + (x_2 - x_1) * alphas).long()
        ys = torch.round(y_1 + (y_2 - y_1) * alphas).long()
        features = image_x[0, :, ys, xs]
        return features.mean(-1).transpose(1, 0)
    
    def aggregate(self, x, left_edges, right_edges, image_x, connections):
        left_x = torch.zeros(x.shape).cuda()
        left_x.index_add_(0, left_edges[:, 0], x[left_edges[:, 1]])
        right_x = torch.zeros(x.shape).cuda()
        right_x.index_add_(0, right_edges[:, 0], x[right_edges[:, 1]])
        global_x = self.image_features(image_x, connections)
        # global_x[:] = x.sum(0)
        x = torch.cat([x, left_x, right_x, global_x], dim=1)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            continue

        return nn.Sequential(*layers)
    
    def forward(self, image, connections, left_edges, right_edges):
        image_x_1 = self.image_layer_1(self.image_layer_0(image))
        image_x_2 = self.image_layer_2(image_x_1)
        image_x_3 = self.image_layer_3(image_x_2)
        image_x_4 = self.image_layer_4(image_x_3)                        
        x = self.layer_1(connections)
        x = self.aggregate(x, left_edges, right_edges, image_x_1, connections)
        x = self.layer_2(x)
        x = self.aggregate(x, left_edges, right_edges, image_x_2, connections)            
        x = self.layer_3(x)
        x = self.aggregate(x, left_edges, right_edges, image_x_3, connections)            
        x = self.layer_4(x)
        x = self.aggregate(x, left_edges, right_edges, image_x_4, connections)            
        connection_pred = self.connection_pred(x)
        return torch.sigmoid(connection_pred).view(-1)
