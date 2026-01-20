import dagster as dg
from .defs.resources import PlateImagingResource,OnDiskArrayResource
from .defs.assets.raw_images import raw_nuclear_channel, raw_cyto_channel
from .defs.assets.processed_images import gaussian_blur_nuclear_channel, clahe_corrected_nuclear_channel, gaussian_blur_cyto_channel,clahe_corrected_cyto_channel
from .defs.assets.generate_cell_object import nuclear_mask, nuclear_labels, nuclear_seeds, nucleus_seeded_cyto_segmentation, cell_clump_corrected_cyto_segmentation

defs = dg.Definitions(
    assets=[raw_nuclear_channel, gaussian_blur_nuclear_channel,clahe_corrected_nuclear_channel,raw_cyto_channel, gaussian_blur_cyto_channel,clahe_corrected_cyto_channel,nuclear_mask, nuclear_labels, nuclear_seeds,nucleus_seeded_cyto_segmentation,cell_clump_corrected_cyto_segmentation],
    resources={
        "imaging_resource": PlateImagingResource(
            base_path="data/",
            mapping_config_path="config/plate_mapping.csv"
        ),
        "array_ressource":OnDiskArrayResource(
            base_dir="data/"
        )
    }
)
