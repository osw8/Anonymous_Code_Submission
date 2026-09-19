

import torch
from torch import nn
import torch
from torch import nn

class DeepConvNet(nn.Module):
    def __init__(self,
                 number_channel: int = 22,
                 nb_classes:    int = 4,
                 dropout_rate: float = 0.5,
                 data_length:  int = 128,   
                 sampling_rate: int = 128,  
                 base_kern:     int = 5,    
                 base_pool:     int = 2):   

        super().__init__()

        if sampling_rate == 128:
            k_t = 5
            p_t = 2
            stride_t = p_t
        elif sampling_rate == 256:
            k_t = 10
            p_t = 3
            stride_t = p_t
        elif sampling_rate == 250:
            k_t = 10
            p_t = 3
            stride_t = p_t
        else: 
            
            fs_ratio = sampling_rate / 128.0

            
            k_t      = max(1, int(round(base_kern  * fs_ratio)))
            p_t      = max(1, int(round(base_pool  * fs_ratio)))
            stride_t = p_t

        self.deepnet = nn.Sequential(
            
            nn.Conv2d(1, 25, (1, k_t), (1, 1), padding="same"),
            nn.Conv2d(25, 25, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, p_t), (1, stride_t)),
            nn.Dropout(dropout_rate),

            
            nn.Conv2d(25, 50, (1, k_t), (1, 1), padding="same"),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, p_t), (1, stride_t)),
            nn.Dropout(dropout_rate),

            
            nn.Conv2d(50, 100, (1, k_t), (1, 1), padding="same"),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, p_t), (1, stride_t)),
            nn.Dropout(dropout_rate),

            
            nn.Conv2d(100, 200, (1, k_t), (1, 1), padding="same"),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, p_t), (1, stride_t)),
            nn.Dropout(dropout_rate),
        )

        
        self.flatten = nn.Flatten()
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1, number_channel, data_length)
            print("dummy shape", dummy.shape)
            feat  = self.deepnet(dummy)
            self.num_features = feat.numel()

        self.classifier = nn.Linear(self.num_features, nb_classes)

    def forward(self, x):
        
        x = x.unsqueeze(1)      
        x = self.deepnet(x)
        x = self.flatten(x)
        return self.classifier(x)
