import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        # x + net(x) is the "Residual" connection
        return self.act(x + self.net(x))
    

class GeoguessrModel(nn.Module):
    def __init__(self, n_classes, model_name='tiny_vit_21m_512.dist_in22k_ft_in1k', pretrained=True):
        super().__init__()
        print(f"Loading backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # global_pool=''
        
        self.embed_dim = self.backbone.num_features 
        
        self.attention_pool = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # unfreeze last layer
        # print("Unfreezing last transformer block for fine-tuning...")
        # for param in self.backbone.blocks[-1].parameters():
        #     param.requires_grad = True
        # for param in self.backbone.norm.parameters():
        #     param.requires_grad = True


        hidden_dim = 2048
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Deep Layer 1
            ResBlock(hidden_dim, dropout=0.2),
            
            # Deep Layer 2
            ResBlock(hidden_dim, dropout=0.2),
            
            # Output Layer
            nn.Linear(hidden_dim, n_classes)
        )

        self.climate_head = nn.Linear(self.embed_dim, 31)  # 31 climate classes
        self.land_cover_head = nn.Linear(self.embed_dim, 11)  # 11 classes
        self.soil_head = nn.Linear(self.embed_dim, 15)  # 15 classes
        self.month_head = nn.Linear(self.embed_dim, 12)  # 12 classes
        
        print(f"Model initialized. Backbone frozen. Head input dim: {self.embed_dim}, Output classes: {n_classes}")

    def forward(self, x):
        features = self.backbone.forward_features(x)

        b_size = features.shape[0]
        query = self.query_token.expand(b_size, -1, -1)
        attn_output, _ = self.attention_pool(query, features, features)
        pooled_features = attn_output.squeeze(1)

        logits_cluster = self.classifier(pooled_features)

        logits_climate = self.climate_head(pooled_features)
        logits_land = self.land_cover_head(pooled_features)
        logits_soil = self.soil_head(pooled_features)
        logits_month = self.month_head(pooled_features)

        return logits_cluster, logits_climate, logits_land, logits_soil, logits_month
    
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.backbone.eval()
            # self.backbone.blocks[-1].train()
            # self.backbone.norm.train()
        return self

    def get_config(self):
        """Helper to get the image transforms required by this specific backbone"""
        return timm.data.resolve_model_data_config(self.backbone)
    
