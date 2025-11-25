import timm
import torch
import torch.nn as nn
import timm

class GeoguessrModel(nn.Module):
    def __init__(self, num_classes, model_name='vit_large_patch14_clip_336.laion2b_ft_in12k_in1k', pretrained=True):
        super().__init__()
        print(f"Loading backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # unfreeze last layer
        print("Unfreezing last transformer block for fine-tuning...")
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

            
        self.input_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes)
        )
        
        print(f"Model initialized. Backbone frozen. Head input dim: {self.input_dim}, Output classes: {num_classes}")

    def forward(self, x):
        features = self.backbone(x)        
        logits = self.classifier(features)
        return logits
    
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.backbone.eval()
            # last layer train
            self.backbone.blocks[-1].train()
            self.backbone.norm.train()
        return self

    def get_config(self):
        """Helper to get the image transforms required by this specific backbone"""
        return timm.data.resolve_model_data_config(self.backbone)
    
