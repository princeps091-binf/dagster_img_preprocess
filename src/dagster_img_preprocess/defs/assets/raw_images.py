import dagster as dg
from ..resources import PlateImagingResource
from ..partitions import imaging_partitions
from typing import Dict, Any
import numpy as np
@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Load raw microscopy images"}
)
def raw_nuclear_channel(
    context: dg.AssetExecutionContext,
    imaging_resource: PlateImagingResource
) -> np.ndarray:
    """Load raw images for a specific plate/well/FOV"""
    
    fov_id = context.partition_key
    
    # Check if this combination exists
    
    # Load all channels for this FOV
    raw_nuclear_channel = imaging_resource.load_raw_channel(fov_id,'C01')
    
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C01',
        "img_size": raw_nuclear_channel.shape[0],
        "saturation_lvl": int((raw_nuclear_channel == raw_nuclear_channel.max()).sum())
    })
    
    return raw_nuclear_channel

@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Load raw microscopy images"}
)
def raw_cyto_channel(
    context: dg.AssetExecutionContext,
    imaging_resource: PlateImagingResource
) -> np.ndarray:
    """Load raw images for a specific plate/well/FOV"""
    
    fov_id = context.partition_key
    
    # Check if this combination exists
    
    # Load all channels for this FOV
    raw_cyto_channel = imaging_resource.load_raw_channel(fov_id,'C02')
    
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C02',
        "img_size": raw_cyto_channel.shape[0],
        "saturation_lvl": int((raw_cyto_channel == raw_cyto_channel.max()).sum())
    })
    
    return raw_cyto_channel
