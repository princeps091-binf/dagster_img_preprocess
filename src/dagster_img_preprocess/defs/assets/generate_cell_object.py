import dagster as dg
import numpy as np
from ..resources import PlateImagingResource,OnDiskArrayResource
from ..partitions import imaging_partitions
from ..utils.mask_creation import create_object_mask,nucleus_watershed_segmentation,nuclear_seed_detection,generate_nucleus_seeded_cyto_mask,correct_cell_segmentation_for_cell_clumping


@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Produce nuclear mask from nucleus channel"},
    deps=["clahe_corrected_nuclear_channel"]

)
def nuclear_mask(
    context: dg.AssetExecutionContext,
    clahe_corrected_nuclear_channel: np.ndarray
) -> np.ndarray:
    """Produce nuclear mask using simple Otsu thresholding on nuclear channel"""
    
    fov_id = context.partition_key
    nuclear_mask = create_object_mask(clahe_corrected_nuclear_channel,60/10)

   
    # Load all channels for this FOV
    
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C01',
    })
    
    return nuclear_mask

@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Produce nuclei labels from nucleus mask"},
    deps=["nuclear_mask"]

)
def nuclear_labels(
    context: dg.AssetExecutionContext,
    array_ressource: OnDiskArrayResource,

    nuclear_mask: np.ndarray
) -> np.ndarray:
    """Create nuclei object based on nucleus mask and watershed segmentation"""
    
    fov_id = context.partition_key
    nuclear_labels = nucleus_watershed_segmentation(nuclear_mask,60/4)

   
    # Load all channels for this FOV
    
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C01',
    })
    nuclear_segmentation_file = array_ressource.save_array(nuclear_labels,'nuclear_segmentation',fov_id)
    
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C01',
        "out_file":nuclear_segmentation_file
    })

    return nuclear_labels

@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Produce nuclei labels from nucleus mask"},
    deps=["nuclear_mask","nuclear_labels"]

)
def nuclear_seeds(
    context: dg.AssetExecutionContext,
    nuclear_mask: np.ndarray,
    nuclear_labels: np.ndarray
) -> np.ndarray:
    """Produce nuclear seeds to use for cytoplasm segmentation"""
    
    fov_id = context.partition_key
    nuclear_seeds = nuclear_seed_detection(nuclear_mask,nuclear_labels,200/4)

   
    # Load all channels for this FOV
    
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C01',
    })
    
    return nuclear_seeds

@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Produce cell labels from nucleus mask and cytoplasm channel"},
    deps=["nuclear_mask","nuclear_seeds","clahe_corrected_cyto_channel"]

)
def nucleus_seeded_cyto_segmentation(
    context: dg.AssetExecutionContext,
    nuclear_mask: np.ndarray,
    nuclear_seeds: np.ndarray,
    clahe_corrected_cyto_channel: np.ndarray
) -> np.ndarray:
    """Produce cell labels from nucleus mask and cytoplasm channel"""
    fov_id = context.partition_key

    cyto_segmentation = generate_nucleus_seeded_cyto_mask(clahe_corrected_cyto_channel,nuclear_mask,nuclear_seeds)
    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C02',
    })
    
    return cyto_segmentation

@dg.asset(
    partitions_def=imaging_partitions,
    metadata={"description": "Correct the cell mask to disaggregate cell clumps"},
    deps=["nuclear_mask","nuclear_labels","nucleus_seeded_cyto_segmentation","clahe_corrected_cyto_channel"]

)
def cell_clump_corrected_cyto_segmentation(
    context: dg.AssetExecutionContext,
    array_ressource: OnDiskArrayResource,

    nuclear_mask: np.ndarray,
    nuclear_labels: np.ndarray,
    nucleus_seeded_cyto_segmentation: np.ndarray,
    clahe_corrected_cyto_channel: np.ndarray
) -> np.ndarray:
    """Correct the cell mask to disaggregate cell clumps"""
    fov_id = context.partition_key

    corrected_cyto_segmentation = correct_cell_segmentation_for_cell_clumping(clahe_corrected_cyto_channel,nucleus_seeded_cyto_segmentation,nuclear_mask,nuclear_labels)

    cell_segmentation_file = array_ressource.save_array(corrected_cyto_segmentation,'cell_segmentation',fov_id)

    context.add_output_metadata({
        "fov": fov_id,
        "channel": 'C02',
        "out_file":cell_segmentation_file
    })
    
    return corrected_cyto_segmentation
