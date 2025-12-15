import torch.nn as nn
import clip


class CLIPModel(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.device = CONFIG['device']
        self.num_classes = CONFIG['classes']
        self.clip_model, transform = clip.load("ViT-L/14@336px", device=self.device, jit=False)
        self.clip_model.float()

        self.train_transform = transform
        self.eval_transform = transform

        self.vision_encoder = self.clip_model.visual
        self.vision_dim = self.vision_encoder.output_dim

        # freeze backbone
        if not CONFIG['backbone_unfrozen']:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(self.vision_dim, self.num_classes)

    def forward(self, images):
        features = self.vision_encoder(images)
        logits = self.classifier(features)
        return logits
