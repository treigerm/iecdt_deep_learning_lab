import logging
import time
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

import iecdt_lab


def plot_reconstructions(batch, reconstructions, data_stats, max_images=8):
    fig, axes = plt.subplots(2, max_images, figsize=(15, 5))
    batch, reconstructions = batch[:max_images], reconstructions[:max_images]
    for i, (img, recon) in enumerate(zip(batch, reconstructions)):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * data_stats["rgb_std"] + data_stats["rgb_mean"]
        recon = recon.permute(1, 2, 0).cpu().numpy()
        recon = recon * data_stats["rgb_std"] + data_stats["rgb_mean"]
        axes[0, i].imshow(img)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon)
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    fig.tight_layout()
    return fig


def validation(cfg, model, test_data_loader, data_stats):
    model.eval()
    running_mse = 0
    num_batches = len(test_data_loader)
    with torch.no_grad():
        for i, (batch, _) in enumerate(test_data_loader):
            batch = batch.to(cfg.device)
            reconstructions = model(batch)
            running_mse += torch.mean((batch - reconstructions) ** 2).cpu().numpy()

            if i == 0:
                fig = plot_reconstructions(batch, reconstructions, data_stats)

            if cfg.smoke_test and i == 10:
                num_batches = i + 1
                break

    val_mse = running_mse / num_batches
    return fig, val_mse


@hydra.main(version_base=None, config_path="config_ae", config_name="train")
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

    model = iecdt_lab.autoencoder.CNNAutoencoder(latent_dim=cfg.latent_dim)
    model = model.to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        for i, (batch, _) in enumerate(train_data_loader):
            optimizer.zero_grad()

            batch = batch.to(cfg.device)
            start_time = time.time()
            preds = model(batch)
            loss = criterion(preds, batch)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            batch_time = end_time - start_time

            if i % cfg.log_freq == 0:
                logging.info(
                    f"Epoch {epoch}/{cfg.epochs} Batch {i}/{len(train_data_loader)}: Loss={loss.item():.2f} Time={batch_time:.3f}s"
                )
                wandb.log({"loss/train": loss.item()})

            if cfg.smoke_test and i == 50:
                break

        eval_fig, val_mse = validation(cfg, model, val_data_loader, data_stats)
        wandb.log({"predictions": eval_fig, "loss/val": val_mse})

        if cfg.smoke_test:
            break

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
