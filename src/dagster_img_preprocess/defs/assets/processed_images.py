import dagster as dg
import numpy as np
from ..resources import PlateImagingResource
from ..partitions import imaging_partitions
from ..utils.img_correction import apply_gaussian_blur_correction, apply_clahe_contrast_correction
@dg.asset(
    partitions_def=imaging_partitions,
    deps=["raw_nuclear_channel"]
)
def gaussian_blur_nuclear_channel(
    context: dg.AssetExecutionContext,
    imaging_resource: PlateImagingResource,
    raw_nuclear_channel: np.ndarray
) -> np.ndarray:
    """Process nuclear-channel microscopy images"""
    
    if raw_nuclear_channel is None:
        return None
    
    fov_id = context.partition_key
    
    gaussian_corrected_img = apply_gaussian_blur_correction(raw_nuclear_channel,obj_size=60)
    

    fov_id = context.partition_key
    # Save processed result
    gauss_corrected_file = imaging_resource.save_img(gaussian_corrected_img,'nuclear_gaussian_blur',fov_id)

    context.add_output_metadata({
        "processing_type": "gaussian blur",
        "out_file":gauss_corrected_file,
        "fov_id":fov_id
    })
    return gaussian_corrected_img

@dg.asset(
    partitions_def=imaging_partitions,
    deps=["gaussian_blur_nuclear_channel"]
)
def clahe_corrected_nuclear_channel(
    context: dg.AssetExecutionContext,
    imaging_resource: PlateImagingResource,
    gaussian_blur_nuclear_channel: np.ndarray
) -> np.ndarray:
    """Process nuclear-channel microscopy images"""
    
    if gaussian_blur_nuclear_channel is None:
        return None
    
    fov_id = context.partition_key
    
    clahe_corrected_img = apply_clahe_contrast_correction(gaussian_blur_nuclear_channel)
    

    fov_id = context.partition_key
    # Save processed result
    clahe_corrected_file = imaging_resource.save_img(clahe_corrected_img,'nuclear_clahe_blur',fov_id)

    context.add_output_metadata({
        "processing_type": "CLAHE contrast correction",
        "out_file":clahe_corrected_file,
        "fov_id":fov_id
    })
    return clahe_corrected_img

@dg.asset(
    partitions_def=imaging_partitions,
    deps=["raw_cyto_channel"]
)
def gaussian_blur_cyto_channel(
    context: dg.AssetExecutionContext,
    imaging_resource: PlateImagingResource,
    raw_cyto_channel: np.ndarray
) -> np.ndarray:
    """Process nuclear-channel microscopy images"""
    
    if raw_cyto_channel is None:
        return None
    
    fov_id = context.partition_key
    
    gaussian_corrected_img = apply_gaussian_blur_correction(raw_cyto_channel,obj_size=60)
    

    fov_id = context.partition_key
    # Save processed result
    gauss_corrected_file = imaging_resource.save_img(gaussian_corrected_img,'cyto_gaussian_blur',fov_id)

    context.add_output_metadata({
        "processing_type": "gaussian blur",
        "out_file":gauss_corrected_file,
        "fov_id":fov_id
    })
    return gaussian_corrected_img

@dg.asset(
    partitions_def=imaging_partitions,
    deps=["gaussian_blur_cyto_channel"]
)
def clahe_corrected_cyto_channel(
    context: dg.AssetExecutionContext,
    imaging_resource: PlateImagingResource,
    gaussian_blur_cyto_channel: np.ndarray
) -> np.ndarray:
    """Process nuclear-channel microscopy images"""
    
    if gaussian_blur_cyto_channel is None:
        return None
    
    fov_id = context.partition_key
    
    clahe_corrected_img = apply_clahe_contrast_correction(gaussian_blur_cyto_channel)
    
    # Save processed result
    clahe_corrected_file = imaging_resource.save_img(clahe_corrected_img,'cyto_clahe_blur',fov_id)

    context.add_output_metadata({
        "processing_type": "CLAHE contrast correction",
        "out_file":clahe_corrected_file,
        "fov_id":fov_id
    })
    return clahe_corrected_img
