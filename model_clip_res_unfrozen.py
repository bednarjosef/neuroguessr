import torch.nn as nn
import torch, timm

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
    

class ResCLIPModel(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.unfrozen_layers = CONFIG['unfrozen_layers']
        self.device = CONFIG['device']
        self.num_classes = CONFIG['clusters']

        self.backbone = timm.create_model(
            'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k', 
            pretrained=True, 
            num_classes=0,
            global_pool='' 
        )

        # transforms
        self.train_transform = timm.data.create_transform(**self.get_config(), is_training=True, scale=(0.5, 1.0))
        self.eval_transform = timm.data.create_transform(**self.get_config(), is_training=False, scale=(0.5, 1.0))

        # dim
        self.vision_dim = self.backbone.num_features

        # attention pooling
        self.attention_pool = nn.MultiheadAttention(embed_dim=self.vision_dim, num_heads=8, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1, 1, self.vision_dim))

        # classifier
        self.classifier = nn.Sequential(
            ResBlock(self.vision_dim, dropout=0.3),
            ResBlock(self.vision_dim, dropout=0.3),
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.num_classes)
        )

        # freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # unfreeze last layers
        if self.unfrozen_layers > 0:
            print(f"Unfreezing last {self.unfrozen_layers} transformer blocks...")
            for block in self.backbone.blocks[-self.unfrozen_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            
            # always unfreeze last LayerNorm
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, images):
        # extract features
        features = self.backbone(images)

        # pool features
        b_size = features.shape[0]
        query = self.query_token.expand(b_size, -1, -1)

        # output
        attn_output, _ = self.attention_pool(query, features, features)
        pooled_features = attn_output.squeeze(1)

        # classify
        logits = self.classifier(pooled_features)
        return logits

    def get_config(self):
        """Helper to get the image transforms required by this specific backbone"""
        return timm.data.resolve_model_data_config(self.backbone)
    