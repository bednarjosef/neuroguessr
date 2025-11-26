import itertools
import os
import random
from time import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from cluster import Clusterer
from dataset import OSV5MDataset, default_image_transform, make_dataloader, train_image_transform
from evaluator import evaluate
from model import ModelConfig, build_model

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None

CONFIG: Dict = {
    "dataset_root": "/home/josef/data/osv5m",
    "cluster_artifact": "artifacts/clusters.npz",
    "train_split": "train",
    "val_split": "val",
    "num_clusters": 1024,
    "image_size": 224,
    "batch_size": 64,
    "accumulation_steps": 1,
    "num_workers": 12,
    "max_steps": 20000,
    "eval_interval": 1000,
    "checkpoint_interval": 2000,
    "val_max_batches": 200,
    "save_dir": "checkpoints",
    "lr": 5e-5,
    "weight_decay": 0.05,
    "warmup_steps": 1000,
    "country_whitelist": ["US", "CA", "GB", "FR", "DE", "AU", "BR", "MX"],
    "clip_model_name": "ViT-H-14",
    "clip_pretrained": "laion2b_s32b_b79k",
    "dropout": 0.1,
    "freeze_backbone": False,
    "compile": True,
    "amp": True,
    "seed": 17,
    "wandb_project": "neuroguessr",
    "wandb_run_name": None,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_warmup_cosine_lr(step: int, max_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(warmup_steps, 1))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def build_dataloaders(cfg: Dict, clusterer: Clusterer) -> tuple:
    train_tf = train_image_transform(cfg["image_size"])
    val_tf = default_image_transform(cfg["image_size"])

    train_ds = OSV5MDataset(
        root=cfg["dataset_root"],
        split=cfg["train_split"],
        clusterer=clusterer,
        countries=cfg["country_whitelist"],
        transform=train_tf,
    )
    val_ds = OSV5MDataset(
        root=cfg["dataset_root"],
        split=cfg["val_split"],
        clusterer=clusterer,
        countries=cfg["country_whitelist"],
        transform=val_tf,
    )

    train_loader = make_dataloader(train_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    val_loader = make_dataloader(val_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    return train_loader, val_loader


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, cfg: Dict) -> str:
    os.makedirs(cfg["save_dir"], exist_ok=True)
    path = os.path.join(cfg["save_dir"], f"step_{step}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": cfg}, path)
    return path


def maybe_init_wandb(cfg: Dict):
    if wandb is None:
        return None
    run = wandb.init(project=cfg["wandb_project"], name=cfg.get("wandb_run_name"), config=cfg, mode="online")
    return run


def train(cfg: Dict) -> None:
    torch.set_float32_matmul_precision("medium")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clusterer = Clusterer.load(cfg["cluster_artifact"])
    if clusterer.centroids_latlon.shape[0] != cfg["num_clusters"]:
        raise ValueError(f"num_clusters={cfg['num_clusters']} but artifact has {clusterer.centroids_latlon.shape[0]}")

    model_cfg = ModelConfig(
        model_name=cfg["clip_model_name"],
        pretrained=cfg["clip_pretrained"],
        num_classes=cfg["num_clusters"],
        dropout=cfg["dropout"],
        freeze_backbone=cfg["freeze_backbone"],
        compile_model=cfg["compile"],
    )
    model = build_model(model_cfg, device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp_enabled = cfg["amp"] and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    train_loader, val_loader = build_dataloaders(cfg, clusterer)
    lr_lambda = lambda step: linear_warmup_cosine_lr(step, cfg["max_steps"], cfg["warmup_steps"])  # noqa: E731
    run = maybe_init_wandb(cfg)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    while global_step < cfg["max_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        global_step += 1
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits, _ = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss = loss / cfg["accumulation_steps"]

        scaler.scale(loss).backward()
        if global_step % cfg["accumulation_steps"] == 0:
            scale = lr_lambda(global_step)
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * scale
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if run is not None and global_step % 10 == 0:
            wandb.log({"train/loss": loss.item() * cfg["accumulation_steps"], "lr": optimizer.param_groups[0]["lr"]}, step=global_step)

        if global_step % cfg["eval_interval"] == 0:
            start = time()
            metrics = evaluate(model, itertools.islice(iter(val_loader), cfg["val_max_batches"]), clusterer, device)
            elapsed = time() - start
            if run is not None:
                wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                wandb.log({"val/time_sec": elapsed}, step=global_step)
            print(f"step {global_step}: val {metrics} ({elapsed:.1f}s)")
            model.train()

        if cfg["checkpoint_interval"] and global_step % cfg["checkpoint_interval"] == 0:
            ckpt_path = save_checkpoint(model, optimizer, global_step, cfg)
            print(f"checkpoint saved: {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    train(CONFIG)
