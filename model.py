import timm
import torch
import torch.nn as nn
import timm

class GeoguessrModel(nn.Module):
    def __init__(self, num_classes, model_name='vit_large_patch14_clip_336.laion2b_ft_in12k_in1k', pretrained=True):
        super().__init__()
        print(f"Loading backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
            
        self.input_dim = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )
        
        print(f"Model initialized. Backbone frozen. Head input dim: {self.input_dim}, Output classes: {num_classes}")

    def forward(self, x):
        features = self.backbone(x)        
        logits = self.head(features)
        return logits

    def get_config(self):
        """Helper to get the image transforms required by this specific backbone"""
        return timm.data.resolve_model_data_config(self.backbone)
    
