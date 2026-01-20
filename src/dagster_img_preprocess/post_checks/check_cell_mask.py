#%%
import skimage as ski
import numpy as np
import napari
import h5py

#%%
cell_segmentation_path = "../../../data/processed/cell_segmentation/BR00115126_A01_F004.h5"
nuclear_segmentation_path = "../../../data/processed/nuclear_segmentation/BR00115126_A01_F004.h5"

with h5py.File(cell_segmentation_path, 'r') as f:
    cell_mask =  np.array(f['cell_segmentation_BR00115126_A01_F004'])
with h5py.File(nuclear_segmentation_path, 'r') as f:
    nuc_mask =  np.array(f['nuclear_segmentation_BR00115126_A01_F004'])

#%%
img_file = "../../../data/processed/cyto_clahe_blur/BR00115126_A01_F004.tif"
img = ski.io.imread(img_file)
# %%
viewer = napari.Viewer()

viewer.add_image(img, name='Grayscale 2D')

viewer.add_labels(
    cell_mask, 
    name='Cell Segmentation Mask',
    opacity=0.5, # Often useful to see the underlying image
)
viewer.add_labels(
    nuc_mask, 
    name='Nuclear Segmentation Mask',
    opacity=0.5, # Often useful to see the underlying image
)

# %%
