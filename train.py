import torch
import torch.optim as optim
import torch.nn as nn

from model import GeoGuessrViT
from clusters import get_clusters, get_cluster_labels
from dataset import create_dataloader
from evaluator import Evaluator
import wandb

tar_directory = 'osv5m_local'
val_directory = 'val_cache_3'

countries = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]

CONFIG = {
    'device': 'cuda',
    'cache_dir': 'cache',
    'eval_interval': 100,
    'countries': countries,
    'num_countries': len(countries),
    'steps': 1000,
    'max_lr_backbone': 5e-6,
    'max_lr_head': 1e-4,
    'batch_size': 256,
    'accum_steps': 1,
    'clusters': 200,
    'sigma_km': 300
}


def train():
    # init wandb
    wandb.init(
        project="neuroguessr",
        config=CONFIG
    )

    torch.set_float32_matmul_precision('high')

    # init clusters
    cluster_centers = get_clusters(CONFIG)
    cluster_labels = get_cluster_labels(CONFIG, cluster_centers)
    cluster_labels = cluster_labels.to(CONFIG['device'])

    # init model
    model = GeoGuessrViT(CONFIG).to(CONFIG['device'])
    model = torch.compile(model)
    transform = model.transform

    # data loader
    train_loader = create_dataloader(CONFIG, tar_directory, cluster_centers, transform, workers=12)

    # evaluator
    evaluator = Evaluator(CONFIG, cluster_centers, transform, val_directory)

    # optimizer
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # adjust these conditions to your model
        if "fc" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG['max_lr_backbone']},
            {"params": head_params, "lr": CONFIG['max_lr_head']},
        ],
        weight_decay=0.01,
    )

    # lr scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[CONFIG['max_lr_backbone'], CONFIG['max_lr_head']],
        steps_per_epoch=CONFIG['steps'],
        epochs=1,
    )

    criterion_cluster = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # train
    model.train()
    optimizer.zero_grad()
    best_median_km = float('inf')
    for step, batch in enumerate(train_loader):
        if step >= (CONFIG['steps'] * CONFIG['accum_steps']):
            break
        images = batch[0].to(CONFIG['device'], non_blocking=True)
        cluster_label = batch[1].to(CONFIG['device'], non_blocking=True)
        target_probs = cluster_labels[cluster_label]

        with torch.amp.autocast('cuda'):
            predicted_clusters = model(images)
            loss_clusters = criterion_cluster(predicted_clusters, target_probs)
            loss = loss_clusters / CONFIG['accum_steps']

        scaler.scale(loss).backward()

        # update weights, log
        if (step + 1) % CONFIG['accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            curr_lr_backbone = scheduler.get_last_lr()[0]
            curr_lr_head = scheduler.get_last_lr()[1]
            loss_val = loss.item() * CONFIG['accum_steps']

            wandb.log({
                "loss/total": loss_val,
                "lr/backbone": curr_lr_backbone,
                "lr/head": curr_lr_head,
            }, step=step+1)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{CONFIG['steps'] * CONFIG['accum_steps']} | Loss: {loss_val:.6f}")
            
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
                    torch.save(model.state_dict(), "geoguessr_best.pth")
                    print(f"🔥 New Best Model! Median Error: {best_median_km:.0f} km")
    

    torch.save(model.state_dict(), "geoguessr_new.pth")
    wandb.finish()
    print("Training Complete.")


if __name__ == '__main__':
    train()