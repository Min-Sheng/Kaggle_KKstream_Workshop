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