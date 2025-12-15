import numpy as np
import torch
import torch.optim as optim
import wandb

from model_clip import CLIPModel
from h3_classification import H3Classifier
from evaluator import Evaluator
from loss import PIGEONLoss
from dataset import create_streetview_dataloader

ds_dir = 'josefbednar/streetview-acw-300k'
val_directory = 'josefbednar/streetview-acw-300k'

countries = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]

CONFIG = {
    'device': 'cuda',
    'cache_dir': 'cache',
    'eval_interval': 100,
    'countries': countries,
    'num_countries': len(countries),
    'steps': 2000,
    'lr_head': 1e-3,
    'lr_group_decay': 0.9,
    'weight_decay': 0.05,
    'batch_size': 512,
    'accum_steps': 1,
    'classes': 0,
    'tau_km': 75,
    'model': 'ViT-L/14@336px',
    'unfrozen_layers': 2,
    'unfreeze_after': -10,
    'backbone_unfrozen': True,
    'h3_resolution': 2,
    'h3_mappings': 'h3_utils/h3_to_class_res2_min200_ring20.json',
    'h3_counts': 'h3_utils/h3_counts_res2.json',
}


def load_model(ckpt_path, model, device):
    raw_state = torch.load(ckpt_path, map_location=device)
    fixed_state = {}
    for k, v in raw_state.items():
        new_k = k
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        fixed_state[new_k] = v

    res = model.load_state_dict(fixed_state, strict=False)
    print("Missing:", res.missing_keys)
    print("Unexpected:", res.unexpected_keys)
    return model


def get_grouped_params(CONFIG, model):
    head_lr = CONFIG['lr_head']
    decay_factor = CONFIG['lr_group_decay']

    param_groups = []
    
    # Classifier Head (Always Highest LR)
    param_groups.append({
        'params': [p for p in model.classifier.parameters() if p.requires_grad], 
        'lr': head_lr
    })
    
    visual = model.vision_encoder

    # Group 1: Output Layers (ln_post, proj)
    post_params = [p for p in visual.ln_post.parameters() if p.requires_grad]
    if visual.proj is not None and visual.proj.requires_grad:
        post_params.append(visual.proj)
        
    param_groups.append({'params': post_params, 'lr': head_lr * decay_factor})
    
    # Group 2: Transformer Blocks (Decaying LR)
    blocks = list(visual.transformer.resblocks)
    for i, block in enumerate(reversed(blocks)):
        current_lr = head_lr * (decay_factor ** (i + 2))
        param_groups.append({
            'params': [p for p in block.parameters() if p.requires_grad],
            'lr': current_lr
        })
        
    # Group 3: Input/Stem (conv1, ln_pre, embeddings)
    stem_params = []
    stem_params.extend([p for p in visual.conv1.parameters() if p.requires_grad])
    stem_params.extend([p for p in visual.ln_pre.parameters() if p.requires_grad])
    if visual.class_embedding.requires_grad:
        stem_params.append(visual.class_embedding)
    if visual.positional_embedding.requires_grad:
        stem_params.append(visual.positional_embedding)
        
    stem_lr = head_lr * (decay_factor ** (len(blocks) + 2))
    param_groups.append({'params': stem_params, 'lr': stem_lr})

    return param_groups

def train():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    h3_classifier = H3Classifier(CONFIG)
    CONFIG['classes'] = h3_classifier.NUM_CLASSES

    # init model
    print('Initializing model...')
    model = CLIPModel(CONFIG).to(CONFIG['device'])

    # load from checkpoint
    # model = load_model('models/neuroguessr-1024-large-osv-pretrained.pth', model, CONFIG['device'])

    model = torch.compile(model)

    train_transform = model.train_transform
    eval_transform = model.eval_transform

    # data loader
    print('Initializing data loader...')
    train_loader = create_streetview_dataloader(CONFIG, h3_classifier, ds_dir, 'train', train_transform, workers=12)

    # evaluator
    print('Initializing evaluator...')
    evaluator = Evaluator(CONFIG, h3_classifier, eval_transform, val_directory)

    # optimizer
    # backbone_params = []
    # head_params = []

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue

    #     if 'backbone' in name:
    #         backbone_params.append(param)
    #     else:
    #         head_params.append(param)

    # optimizer = optim.AdamW(
    #     [
    #         {"params": backbone_params, "lr": CONFIG['max_lr_backbone']},
    #         {"params": head_params, "lr": CONFIG['max_lr_head']},
    #     ],
    #     weight_decay=0.05,
    # )

    param_groups = get_grouped_params(CONFIG, model)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])

    print('Initializing scheduler...')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["steps"],
    )

    print('Initializing PIGEON loss...')
    pigeon = PIGEONLoss(CONFIG)
    scaler = torch.amp.GradScaler('cuda')

    # init wandb
    wandb.init(
        project="neuroguessr",
        config=CONFIG
    )

    # train
    print('Beginning training...')
    model.train()
    optimizer.zero_grad()
    best_median_km = float('inf')
    seen_clusters = np.zeros(CONFIG['classes'], dtype=bool)

    for step, batch in enumerate(train_loader):
        if step >= (CONFIG['steps'] * CONFIG['accum_steps']):
            break

        # if step < (CONFIG['unfreeze_after'] + 1):
        #     model.backbone.eval() # Tell backbone to act frozen (norm layers)
        #     for p in backbone_params:
        #         p.requires_grad = False
        # elif step == (CONFIG['unfreeze_after'] + 1):
        #     print("❄️ -> 🔥 Unfreezing Backbone now!")
        #     model.backbone.train()
        #     for p in backbone_params:
        #         p.requires_grad = True

        images = batch[0].to(CONFIG['device'], non_blocking=True)
        true_clusters = batch[1].to(CONFIG['device'], non_blocking=True)
        true_lats = batch[2].to(CONFIG['device'], non_blocking=True)
        true_lons = batch[3].to(CONFIG['device'], non_blocking=True)

        seen_clusters[true_clusters.cpu().numpy()] = True

        with torch.amp.autocast('cuda'):
            logits = model(images)
            smooth_loss = pigeon.loss(CONFIG, logits, true_clusters, true_lats, true_lons)
            smooth_loss = smooth_loss / CONFIG['accum_steps']

        scaler.scale(smooth_loss).backward()

        if (step + 1) % CONFIG['accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            curr_lr_backbone = scheduler.get_last_lr()[0]
            curr_lr_head = scheduler.get_last_lr()[1]
            loss_val = smooth_loss.item() * CONFIG['accum_steps']

            with torch.no_grad():  # TODO: softmax?
                preds = logits.argmax(dim=1)
                train_acc = (preds == true_clusters).float().mean().item() * 100
            
            wandb.log({
                "loss/total": loss_val,
                "lr/backbone": curr_lr_backbone,
                "lr/head": curr_lr_head,
                "train/acc_top1": train_acc,
            }, step=step+1)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{CONFIG['steps'] * CONFIG['accum_steps']} | Loss: {loss_val:.6f} | Train acc: {train_acc:.2f}%")
            
            # evaluate
            if evaluator and (((step + 1) % CONFIG['eval_interval'] == 0) or step == 0):
                metrics = evaluator.run(model)
                wandb.log(metrics, step=step+1)

                current_median = metrics['val/top1_median_km']
                
                # save latest
                torch.save(model.state_dict(), f"checkpoint_last.pth")
                
                # save best
                if current_median < best_median_km:
                    best_median_km = current_median
                    torch.save(model.state_dict(), "neuroguessr-1024-large-streetview-h3-best.pth")
                    print(f"🔥 New Best Model! Median Error: {best_median_km:.0f} km")
    

    torch.save(model.state_dict(), "neuroguessr-1024-large-streetview-h3-final.pth")
    wandb.finish()
    print("Training Complete.")


if __name__ == '__main__':
    train()