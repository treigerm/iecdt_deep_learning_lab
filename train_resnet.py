import logging
import os
import random
import ssl
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

import iecdt_lab

# Necessary to download pre-trained weights.
ssl._create_default_https_context = ssl._create_unverified_context


def plot_images(batch, cloud_fraction, preds, data_stats, max_images=8):
    fig, axes = plt.subplots(1, max_images, figsize=(15, 5))
    batch, cloud_fraction, preds = (
        batch[:max_images],
        cloud_fraction[:max_images],
        preds[:max_images],
    )
    for i, (img, true_cf, pred_cf) in enumerate(zip(batch, cloud_fraction, preds)):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * data_stats["rgb_std"] + data_stats["rgb_mean"]
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_cf:.2f}\nPred: {pred_cf:.2f}")
        axes[i].axis("off")

    fig.tight_layout()
    return fig


def validation(cfg, model, test_data_loader, data_stats):
    model.eval()
    running_mse = 0
    num_batches = len(test_data_loader)
    with torch.no_grad():
        for i, (batch, labels) in enumerate(test_data_loader):
            _, cloud_fraction, _, _ = labels
            batch = batch.to(cfg.device)
            cloud_fraction = cloud_fraction.float().to(cfg.device)
            preds = model(batch).flatten()
            running_mse += torch.mean((cloud_fraction - preds) ** 2).cpu().numpy()

            if i == 0:
                fig = plot_images(
                    batch, cloud_fraction.cpu().numpy(), preds, data_stats
                )

            if cfg.smoke_test and i == 10:
                num_batches = i + 1
                break

    val_mse = running_mse / num_batches
    return fig, val_mse


@hydra.main(version_base=None, config_path="config_resnet", config_name="train")
def main(cfg: DictConfig):
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    wandb.login(key=os.environ["WANDB_API_KEY"])
    # Generate ID to store and resume run.
    wandb_id = wandb.util.generate_id()
    wandb.init(
        id=wandb_id,
        resume="allow",
        project=cfg.wandb.project,
        group=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
    )

    data_stats = np.load(cfg.train_rgb_stats)
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # Normalize the data.
            torchvision.transforms.Normalize(
                mean=data_stats["rgb_mean"], std=data_stats["rgb_std"]
            ),
        ]
    )
    train_data_loader, val_data_loader = iecdt_lab.data_loader.get_data_loaders(
        tiles_path=cfg.tiles_path,
        train_metadata=cfg.train_metadata,
        val_metadata=cfg.val_metadata,
        batch_size=cfg.batch_size,
        data_transforms=data_transforms,
        dataloader_workers=cfg.dataloader_workers,
        load_tiles=cfg.load_tiles,
    )

    model = torchvision.models.resnet18(
        weights="IMAGENET1K_V1" if cfg.pretrained else None
    )
    if cfg.pretrained and not cfg.finetune:
        # Ensure we don't calculate gradients for the pre-trained weights.
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid(),  # Ensure that output lies between 0 and 1.
    )
    model = model.to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters() if cfg.finetune else model.fc.parameters(),
        lr=cfg.learning_rate,
    )

    for epoch in range(cfg.epochs):
        model.train()
        for i, (batch, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()

            _, cloud_fraction, _, _ = labels
            batch = batch.to(cfg.device)
            cloud_fraction = cloud_fraction.float().to(cfg.device)
            start_time = time.time()
            preds = model(batch)  # (num_batch, 1)
            loss = criterion(preds.flatten(), cloud_fraction)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            batch_time = end_time - start_time

            if i % cfg.log_freq == 0:
                logging.info(
                    f"Epoch {epoch}/{cfg.epochs} Batch {i}/{len(train_data_loader)}: Loss={loss.item():.2f} Time={batch_time:.2f}s"
                )
                wandb.log({"loss/train": loss.item()})

            if cfg.smoke_test and i == 50:
                break

        eval_fig, val_mse = validation(cfg, model, val_data_loader, data_stats)
        wandb.log({"predictions": eval_fig, "val_mse": val_mse})

        if cfg.smoke_test:
            break

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
