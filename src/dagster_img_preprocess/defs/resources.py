
import dagster as dg
import pandas as pd
import skimage as ski
from pathlib import Path
import numpy as np
import h5py
from typing import Optional
import os

class PlateImagingResource(dg.ConfigurableResource):
    """Resource for handling microscopy images with plate/well/FOV/channel structure"""
    
    base_path: str = "data"
    mapping_config_path: str = "config/plate_mapping.csv"
    
    @property
    def plate_mapping(self) -> pd.DataFrame:
        if not hasattr(self, '_plate_mapping_cache'):
            self._plate_mapping_cache = pd.read_csv(self.mapping_config_path)
        return self._plate_mapping_cache

    def get_image_path(self, filename: str) -> Path:
        """Get full path to an image file in the data folder"""
        return Path(self.base_path) / filename
    
    def get_all_partition_combinations(self) -> pd.Series:
        """Extract all valid partition combinations from CSV"""
        df = self.plate_mapping
        
        # Get unique combinations of partition keys
        unique_combinations = df[['fov_id']].drop_duplicates()
        
        return unique_combinations
    
    def load_raw_channel(self, fov_id: str, channel: str) -> np.ndarray:
        """Load a single channel image"""
        img_file = (self.plate_mapping
                        .query("fov_id == @fov_id and channel == @channel")
                        .file.to_list()[0])
        img_path = self.get_image_path(f"raw/{img_file}")
        return ski.io.imread(img_path)

    def load_img(self,op_id: str, fov_id: str, channel: str) -> np.ndarray:
        """Load a single channel image"""
        img_file = f"{op_id}/{fov_id}.tif"
        img_path = self.get_image_path(img_file)
        return ski.io.imread(img_path)


    def save_img(self, img: np.ndarray,img_op: str,fov_id: str) -> str:
        """Save a channel image"""
        path = f"{self.base_path}/processed/{img_op}/{fov_id}.tif"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ski.io.imsave(path, img)
        return path


class OnDiskArrayResource(dg.ConfigurableResource):
    """Resource for manual HDF5 file operations."""
    
    base_dir: str = "data"
    compression: Optional[str] = "gzip"
    compression_level: Optional[int] = 4
    
    def _get_path(self, filename: str) -> str:
        """Generate full file path."""
        if not filename.endswith('.h5'):
            filename += '.h5'
        return os.path.join(self.base_dir, filename)
    
    def save_array(self, 
                   array: np.ndarray, 
                   img_op: str,
                   fov_id: str) -> str:
        """Save NumPy array to HDF5 file."""
        file_path = f"{self.base_dir}/processed/{img_op}/{fov_id}.h5"
        dset_name = f"{img_op}_{fov_id}"
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        
        with h5py.File(file_path, 'w') as f:
            if self.compression:
                f.create_dataset(
                    dset_name,
                    data=array,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            else:
                f.create_dataset(dset_name, data=array)
            
            # Add metadata
            f.attrs['dtype'] = str(array.dtype)
            f.attrs['shape'] = array.shape
        
        return file_path
    
    def load_array(self, 
                   op_id: str, 
                   fov_id: str) -> np.ndarray:
        """Load NumPy array from HDF5 file."""
        file_path = f"{self.base_dir}/processed/{op_id}/{fov_id}.h5"
        dset_name = f"{op_id}_{fov_id}"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        
        with h5py.File(file_path, 'r') as f:
            return np.array(f[dset_name])
    