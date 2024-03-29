import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class KKstreamModel(BaseModel):
    def __init__(self, num_classes=28):
        super(KKstreamModel, self).__init__()
        self.fc1 = nn.Linear(896, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256,512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512,1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc6 = nn.Linear(1024,num_classes)
        self.ativ = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        #x = F.dropout(x, p=0.75, training=self.training)
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        #x = F.dropout(x, p=0.75, training=self.training)
        x = self.bn4(x)
        x = F.relu(self.fc5(x))
        x = self.bn5(x)
        x = self.ativ(self.fc6(x))
        return x

# 1x7 convolution
def conv1x7(in_channels, out_channels, stride=1, padding=3, bias=False):
    return nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                     stride=stride, padding=padding, bias=bias)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1x7(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv1x7(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out

# ResNet
class KKstreamModelResNet(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2,2,2,2,2,2], num_classes=28):
        super(KKstreamModelResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = conv1x7(23, 16, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        #self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2) #small
        self.layer4 = self.make_layer(block, 128, layers[3], 2) #big
        self.layer5 = self.make_layer(block, 256, layers[4], 2) #large
        self.layer6 = self.make_layer(block, 512, layers[5], 2) #larger
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.ativ = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x7(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #out = x[:, None]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.ativ(out)
        return out

# RNN
class KKstreamModelRNN(nn.Module):
    def __init__(self, num_classes=28):
        super(KKstreamModelRNN, self).__init__()
        #self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.gru = nn.GRU(1, 128, num_layers=2, dropout=0.375, bidirectional=False)
        self.fc1 = nn.Linear(128*896, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 28)
        self.ativ = nn.Sigmoid()
    
    def forward(self, x):
        
        # Expand channel dimension to (batch_size, seq_len, input_size) for RNN with batch_first 
        out = x[:, : , None]
        out, _ = self.gru(out)
        # Flatten the dimension from (batch_size, seq_len, hidden_size) to (batch_size, seq_len x hidden_size)
        out = out.view(out.size(0), -1)
        out = F.elu(self.fc1(out))
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.ativ(out)
        return out
    
# RNN
class KKstreamModelRNNCNN(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2,2,2,2,2,2], num_classes=28):
        super(KKstreamModelRNNCNN, self).__init__()
        self.in_channels = 16
        self.bi_gru = nn.GRU(23, 128, num_layers=1, bidirectional=True)
        self.conv1 = conv1x7(256, 16, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.elu = nn.ELU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2) #small
        self.layer4 = self.make_layer(block, 128, layers[3], 2) #big
        self.layer5 = self.make_layer(block, 256, layers[4], 2) #large
        self.layer6 = self.make_layer(block, 512, layers[5], 2) #large
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.ativ = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x7(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        # Expand channel dimension to (batch_size, input_size, seq_len) for CNN
        #out = x[:, None]
        # Turn (batch_size, hidden_size, seq_len) back into (seq_len, batch_size, hidden_size) for RNN
        out = x.transpose(1, 2).transpose(0, 1)
        out, hidden = self.bi_gru(out)

        # Turn (seq_len, batch_size, hidden_size) into (batch_size, input_size, seq_len) for CNN
        out = out.transpose(0, 1).transpose(1, 2)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.ativ(out)
        return out

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, bias=bias)

# Residual block
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out

# RNN with 2D CNN
class KKstreamModelRNNCNN2D(nn.Module):
    def __init__(self, block=ResidualBlock2D, layers=[2,2,2,2,2,2], num_classes=28):
        super(KKstreamModelRNNCNN2D, self).__init__()
        self.in_channels = 16
        self.bi_gru = nn.GRU(23, 64, num_layers=1, bidirectional=False)
        self.conv1 = conv3x3(64, 16, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.elu = nn.ELU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2) #small
        self.layer4 = self.make_layer(block, 128, layers[3], 2) #big
        #self.layer5 = self.make_layer(block, 256, layers[4], 2) #large
        #self.layer6 = self.make_layer(block, 512, layers[5], 2) #large
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.ativ = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        # Turn (batch_size, hidden_size, seq_len) back into (seq_len, batch_size, hidden_size) for RNN
        out = x.transpose(1, 2).transpose(0, 1)
        out, hidden = self.bi_gru(out)
        # Turn (seq_len, batch_size, hidden_size) into (batch_size, input_size, seq_len) for CNN
        out = out.transpose(0, 1).transpose(1, 2)
        out = out[..., None]
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.layer5(out)
        #out = self.layer6(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.ativ(out)
        return out

# RNN
class KKstreamModelRNNCNN2(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2,2,2,2,2,2], num_classes=28):
        super(KKstreamModelRNNCNN2, self).__init__()
        self.in_channels = 16
        self.bi_gru = nn.GRU(23, 128, num_layers=2, dropout=0.375, bidirectional=True)
        self.conv1 = conv1x7(256, 16, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.elu = nn.ELU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2) #small
        self.layer4 = self.make_layer(block, 128, layers[3], 2) #big
        self.layer5 = self.make_layer(block, 256, layers[4], 2) #large
        self.layer6 = self.make_layer(block, 512, layers[5], 2) #large
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.ativ = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x7(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        # Expand channel dimension to (batch_size, input_size, seq_len) for CNN
        #out = x[:, None]
        # Turn (batch_size, hidden_size, seq_len) back into (seq_len, batch_size, hidden_size) for RNN
        out = x.transpose(1, 2).transpose(0, 1)
        out, hidden = self.bi_gru(out)

        # Turn (seq_len, batch_size, hidden_size) into (batch_size, input_size, seq_len) for CNN
        out = out.transpose(0, 1).transpose(1, 2)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.ativ(out)
        return out
    
# RNN
class KKstreamModelRNNCNN3(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2,2,2,2,2,2], num_classes=28):
        super(KKstreamModelRNNCNN3, self).__init__()
        self.in_channels = 16
        self.bi_gru = nn.GRU(23, 128, num_layers=3, dropout=0.375, bidirectional=True)
        self.conv1 = conv1x7(256, 16, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.elu = nn.ELU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2) #small
        self.layer4 = self.make_layer(block, 128, layers[3], 2) #big
        self.layer5 = self.make_layer(block, 256, layers[4], 2) #large
        self.layer6 = self.make_layer(block, 512, layers[5], 2) #large
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.ativ = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x7(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        # Expand channel dimension to (batch_size, input_size, seq_len) for CNN
        #out = x[:, None]
        # Turn (batch_size, hidden_size, seq_len) back into (seq_len, batch_size, hidden_size) for RNN
        out = x.transpose(1, 2).transpose(0, 1)
        out, hidden = self.bi_gru(out)

        # Turn (seq_len, batch_size, hidden_size) into (batch_size, input_size, seq_len) for CNN
        out = out.transpose(0, 1).transpose(1, 2)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.ativ(out)
        return out