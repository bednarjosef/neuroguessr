import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class GeoGuessrViT(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()

        # backbone model
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)
        self.transform = weights.transforms()

        # number of extracted features
        in_features = self.backbone.heads.head.in_features

        # replace the classification layer with my own
        self.backbone.heads.head = nn.Linear(in_features, CONFIG['clusters'])

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = GeoGuessrViT({'clusters': 200})
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(out.shape)
