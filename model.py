import timm

model = timm.create_model('vit_large_patch14_clip_336.laion2b_ft_in12k_in1k', pretrained=True, num_classes=0)
model = model.eval()