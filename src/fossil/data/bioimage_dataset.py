from skimage.filters import threshold_otsu
from typing import Sequence, Literal
from numpy.typing import NDArray
from bioio import BioImage
from pathlib import Path
import numpy as np
import tifffile
import torch
import math
import os


class BioImageMask:
    """A class for creating and querying tile-based masks to identify foreground 
    regions in biological images. Uses Otsu thresholding or provided background 
    values to determine foreground pixels, then aggregates into tiles.
    """

    def __init__(
        self,
        path: os.PathLike,
        tile_size: int = 1024,
        level: int = -1,
        channels: Sequence[int] | None = None,
        min_foreground_fraction: float = 0.10,
        background_values: Sequence[float] | None = None,
    ) -> None:
        """Initialize the BioImageMask.
        
        Parameters
        ----------
        path : os.PathLike
            Path to the biological image file
        tile_size : int, default=1024
            Size of tiles in pixels at full resolution
        level : int, default=-1
            Resolution level to use (-1 for lowest resolution)
        channels : Sequence[int] or None, default=None
            Channels to include in mask (None for all)
        min_foreground_fraction : float, default=0.10
            Minimum fraction of foreground pixels required for a tile to be 
            considered foreground
        background_values : Sequence[float] or None, default=None
            Background threshold values per channel.
        """
        self._path = Path(path)
        self._tile_size = tile_size
        self._level = level
        self._channels = channels
        self._bg_fraction = 1 - min_foreground_fraction
        self._bg_values = background_values

        # Initialize attributes that will be set by _set_tile_mask
        self._tile_mask: NDArray | None = None
        self._foreground_indices: NDArray | None = None
        self._full_res_dims: tuple[int, int] | None = None
        self._downsample: int | None = None
        
        self._set_tile_mask()

    def _set_tile_mask(self) -> None:
        """Generate a tile-based mask indicating which tiles contain foreground."""
        image = BioImage(self._path)
        if self._level < 0:
            self._level += len(image.resolution_levels)
        image.set_resolution_level(self._level)
        if self._channels is None:
            self._channels = list(range(image.dims.C))

        # Determine background thresholds if not provided
        if self._bg_values is None:
            self._bg_values = []
            for channel in self._channels:
                image_data = image.get_image_data('YX', C=channel)
                self._bg_values.append(threshold_otsu(image_data))
        elif len(self._bg_values) != len(self._channels):
            raise ValueError(
                "Background values must have same number of values as "
                "channels."
            )
            
        # Mask pixels greater than background
        mask = np.greater_equal(
            image.get_image_data('YXC', C=self._channels),
            np.array(self._bg_values),
        ).any(axis=-1)
        
        # Get downsample factor and full-resolution dimensions
        lvl_dims = image.resolution_level_dims
        x = image.dims.order.index('X')
        y = image.dims.order.index('Y')
        self._full_res_dims = (lvl_dims[0][y], lvl_dims[0][x]) # (height, width)
        
        x_factor = lvl_dims[0][x] / lvl_dims[self._level][x]
        y_factor = lvl_dims[0][y] / lvl_dims[self._level][y]
        self._downsample = min(round(x_factor), round(y_factor))

        # Aggregate mask according to tile size at the highest-resolution
        tile_size = max(self._tile_size // self._downsample, 1)
        height, width = mask.shape
        
        # Calculate number of tiles
        n_tiles_y = math.ceil(height / tile_size)
        n_tiles_x = math.ceil(width / tile_size)
        
        # Create tile mask
        tile_mask = np.zeros((n_tiles_y, n_tiles_x), dtype=bool)
        
        for tile_y in range(n_tiles_y):
            y_start = tile_y * tile_size
            y_end = min((tile_y + 1) * tile_size, height)
            
            for tile_x in range(n_tiles_x):
                x_start = tile_x * tile_size
                x_end = min((tile_x + 1) * tile_size, width)
                
                # Get tile region
                tile_region = mask[y_start:y_end, x_start:x_end]
                
                # Calculate foreground fraction
                foreground_fraction = np.mean(tile_region)
                
                # Mark tile as foreground if it exceeds threshold
                tile_mask[tile_y, tile_x] = foreground_fraction > \
                                            (1 - self._bg_fraction)
        
        self._tile_mask = tile_mask
        self._foreground_indices = np.argwhere(self._tile_mask)

    def is_foreground(self, x: int, y: int) -> bool:
        """Check if a full-resolution coordinate is in a foreground tile.
        
        Parameters
        ----------
        x : int
            X coordinate at full resolution
        y : int
            Y coordinate at full resolution
            
        Returns
        -------
        bool
            True if the coordinate falls within a foreground tile
        """
        # Convert full-resolution coordinates to tile coordinates
        tile_x = x // self._tile_size
        tile_y = y // self._tile_size
        
        # Check bounds
        if (tile_y < 0 or tile_y >= self._tile_mask.shape[0] or 
            tile_x < 0 or tile_x >= self._tile_mask.shape[1]):
            return False
            
        return self._tile_mask[tile_y, tile_x]

    def is_foreground_window(
        self,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
        mode: Literal['any', 'all'] = "any",
    ) -> bool:
        """Check if a window at full-resolution overlaps foreground tiles.
        
        Parameters
        ----------
        xmin : int
            Minimum X coordinate of window
        xmax : int
            Maximum X coordinate of window  
        ymin : int
            Minimum Y coordinate of window
        ymax : int
            Maximum Y coordinate of window
        mode : {'any', 'all'}, default='any'
            'any' if any overlapping tile should be foreground,
            'all' if all overlapping tiles should be foreground
                      
        Returns
        -------
        bool
            True if window overlaps foreground according to specified mode
        """
        # Convert window coordinates to tile indices
        tile_xmin = max(0, xmin // self._tile_size)
        tile_xmax = min(self._tile_mask.shape[1] - 1, xmax // self._tile_size)
        tile_ymin = max(0, ymin // self._tile_size)
        tile_ymax = min(self._tile_mask.shape[0] - 1, ymax // self._tile_size)
        
        # Extract overlapping tiles
        overlapping_tiles = self._tile_mask[
            tile_ymin:tile_ymax+1,
            tile_xmin:tile_xmax+1
        ]
        
        if overlapping_tiles.size == 0:
            return False
            
        if mode == 'any':
            return np.any(overlapping_tiles)
        elif mode == 'all':
            return np.all(overlapping_tiles)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'any' or 'all'.")

    def get_masked_coordinates(self, n: int) -> NDArray:
        """Get n random (x, y) coordinates from within foreground tiles.

        This is a vectorized function for efficient generation of many 
        coordinates. The coordinates are given at the full, original image 
        resolution.

        Parameters
        ----------
        n : int
            The number of random coordinates to generate.

        Returns
        -------
        NDArray
            A numpy array of shape (n, 2) where each row is an [x, y] 
            coordinate.

        Raises
        ------
        ValueError
            If no foreground tiles are found in the mask.
        """
        if len(self._foreground_indices) == 0:
            raise ValueError("No foreground tiles found in the mask.")

        # Randomly select foreground tiles with replacement
        random_indices = np.random.randint(0, len(self._foreground_indices), 
                                           size=n)
        selected_tiles = self._foreground_indices[random_indices]
        tile_y, tile_x = selected_tiles[:, 0], selected_tiles[:, 1]
        xmins = tile_x * self._tile_size
        ymins = tile_y * self._tile_size
        
        # Clamp max coordinates to actual image dimensions
        img_height, img_width = self._full_res_dims
        xmaxs = np.minimum(xmins + self._tile_size, img_width)
        ymaxs = np.minimum(ymins + self._tile_size, img_height)

        # Generate n random coordinates within the respective bounding boxes
        rand_x = xmins + (np.random.rand(n) * (xmaxs - xmins)).astype(int)
        rand_y = ymins + (np.random.rand(n) * (ymaxs - ymins)).astype(int)

        return np.stack((rand_x, rand_y), axis=1)
    
    def sample_coordinate(self, seed: int) -> tuple[int, int]:
        """Deterministically sample one (x, y) coordinate within a foreground tile.

        Parameters
        ----------
        seed : int
            Random seed to deterministically select the coordinate.

        Returns
        -------
        tuple[int, int]
            A single (x, y) coordinate in full-resolution space.

        Raises
        ------
        ValueError
            If no foreground tiles are found in the mask.
        """
        if len(self._foreground_indices) == 0:
            raise ValueError("No foreground tiles found in the mask.")

        rng = np.random.default_rng(seed)

        # Pick one foreground tile deterministically
        tile_y, tile_x = self._foreground_indices[
            rng.integers(len(self._foreground_indices))
        ]

        # Tile bounds at full resolution
        xmin = tile_x * self._tile_size
        ymin = tile_y * self._tile_size
        xmax = min(xmin + self._tile_size, self._full_res_dims[1])
        ymax = min(ymin + self._tile_size, self._full_res_dims[0])

        # Sample a coordinate within that tile deterministically
        x = rng.integers(xmin, xmax)
        y = rng.integers(ymin, ymax)

        return int(x), int(y)
    

class BioImageDataset(torch.utils.data.Dataset):
    """TODO: Description
    """
    def __init__(
        self,
        path: os.PathLike,
        tile_size: int = 1024,
        mask_kwargs: dict = {},
    ):
        self._path = path
        self._tile_size = tile_size
        self._mask = BioImageMask(self._path, self._tile_size, **mask_kwargs)

        # Image sizes to clip to
        image = BioImage(self._path)
        self._xmax = image.dims.X - self._tile_size
        self._ymax = image.dims.Y - self._tile_size

    def __getitem__(self, index):
        """TODO: Description.
        """
        # Sample random tile
        x, y = self._mask.sample_coordinate(seed=index)
        x = min(self._xmax, x)
        y = min(self._ymax, y)

        # Slice BioImage and return as Tensor
        xs = slice(x, x + self._tile_size)
        ys = slice(y, y + self._tile_size)
        cs = slice(None, None, None)
        image_data = tifffile.imread(
            self._path,
            level=0,
            selection=(cs, ys, xs),
            as_zarr=True,
        )
        return torch.tensor(image_data)
