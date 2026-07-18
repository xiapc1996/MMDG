import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(FinalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, num_classes)
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, domain_feat):
        return self.classifier(domain_feat) 
