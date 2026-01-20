import dagster as dg
from .resources import PlateImagingResource

def create_imaging_partitions(config_path: str = "config/plate_mapping.csv"):
    """Dynamically create partitions from csv configuration"""
    
    # Create temporary resource to read config
    imaging_resource = PlateImagingResource(mapping_config_path=config_path)
    
    # Extract unique values for each dimension
    all_combos = imaging_resource.get_all_partition_combinations()
    
    fovs = all_combos.fov_id.drop_duplicates().sort_values().to_list()
    
    return dg.StaticPartitionsDefinition(fovs)

# Create the partitions
imaging_partitions = create_imaging_partitions()
