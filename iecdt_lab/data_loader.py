import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tqdm
from concurrent.futures import ThreadPoolExecutor


@dataclass
class RGBTile:
    time_ix: int
    lat_ix: slice
    lon_ix: slice
    mean_cloud_lengthscale: float
    cloud_fraction: float
    cloud_iorg: float
    fractal_dimension: float


class GOESRGBTiles(Dataset):
    def __init__(self, tiles_file: str, metadata_file: str, transform=None, load_tiles=False) -> None:
        """
        Args:
            tiles_file (str): Path to a directory containing all the tiles.
            metadata_file (str): Path to a CSV file containing metadata with the header
                "tile_id,time_ix,lat_ix,lon_ix,$metric1,$metric2,...". `lat_ix` and `lon_ix`
                will be strings in the format "slice(start, stop, None)".
            load_tiles (bool): If True, load all tiles into memory. Otherwise, load on demand.
        """
        self.tiles_metadata = self._parse_metadata(metadata_file)
        self.tiles_dir = tiles_file
        self.transform = transform
        self.load_tiles = load_tiles
        if self.load_tiles:
            unique_time_ixs = set([t.time_ix for t in self.tiles_metadata])
            self.tiles = {}
            
            def load_tile(time_ix):
                return time_ix, np.load(f"{self.tiles_dir}/{time_ix}/time_step.npy")
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                for time_ix, tile in tqdm.tqdm(
                    executor.map(load_tile, unique_time_ixs),
                    total=len(unique_time_ixs),
                    desc="Loading tiles into memory"
                ):
                    self.tiles[time_ix] = tile

    def __len__(self) -> int:
        return len(self.tiles_metadata)

    def __getitem__(self, ix: int) -> tuple[np.ndarray, tuple]:
        if self.load_tiles:
            tile = self.tiles[self.tiles_metadata[ix].time_ix]
        else:
            tiles_file = f"{self.tiles_dir}/{self.tiles_metadata[ix].time_ix}/time_step.npy"
            tile = np.load(tiles_file)
        tile = tile[
            self.tiles_metadata[ix].lat_ix,
            self.tiles_metadata[ix].lon_ix,
        ]
        labels = (
            self.tiles_metadata[ix].mean_cloud_lengthscale,
            self.tiles_metadata[ix].cloud_fraction,
            self.tiles_metadata[ix].cloud_iorg,
            self.tiles_metadata[ix].fractal_dimension,
        )
        if self.transform:
            tile = self.transform(tile)

        return tile, labels

    def _parse_metadata(self, metadata_file: str) -> list[RGBTile]:
        metadata = pd.read_csv(metadata_file)
        # Parse each row into an RGBTile object
        metadata = [
            RGBTile(
                time_ix=row["time_ix"],
                lat_ix=self._extract_slice(row["lat_ix"]),
                lon_ix=self._extract_slice(row["lon_ix"]),
                mean_cloud_lengthscale=row["mean_cloud_lengthscale"],
                cloud_fraction=row["cloud_fraction"],
                cloud_iorg=row["cloud_iorg"],
                fractal_dimension=row["fractal_dimension"],
            )
            for _, row in metadata.iterrows()
        ]
        return metadata

    @staticmethod
    def _extract_slice(s: str) -> slice:
        """Takes as input a string of the form 'slice($start, $stop, None)' and
        returns a Python slice object."""
        # NOTE: We could use `eval` here but that's not safe.
        match = re.match(r"slice\((\d+), (\d+), None\)", s)
        if match is None:
            raise ValueError(f"Could not match {s}")
        return slice(int(match.group(1)), int(match.group(2)))


def get_data_loaders(
    tiles_path,
    train_metadata,
    val_metadata,
    batch_size,
    data_transforms,
    dataloader_workers,
    load_tiles=False,
):
    train_ds = GOESRGBTiles(
        tiles_file=tiles_path,
        metadata_file=train_metadata,
        transform=data_transforms,
        load_tiles=load_tiles,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )
    val_ds = GOESRGBTiles(
        tiles_file=tiles_path,
        metadata_file=val_metadata,
        transform=data_transforms,
        load_tiles=load_tiles,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )
    return train_data_loader, val_data_loader
