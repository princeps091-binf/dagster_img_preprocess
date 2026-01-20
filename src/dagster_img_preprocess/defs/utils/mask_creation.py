import skimage as ski
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
import networkx as nx
def create_object_mask(img,noise_size,thresh_coef=1):
    tmp_quant = np.quantile(img,[0.25,0.75])
    tmp_IQR = tmp_quant[1] - tmp_quant[0]
    tmp_n = img.shape[0] * img.shape[1]
    optim_width = 2 * tmp_IQR * tmp_n**(-1/3)
    tmp_nbin = ((np.max(img) - np.min(img))/optim_width).astype(np.uint)
    tmp_thresh = ski.filters.threshold_otsu(img,nbins=tmp_nbin)
    otsu_thresh_img = img > (thresh_coef*tmp_thresh)
    obj_mask = ski.morphology.remove_small_objects(otsu_thresh_img, min_size=noise_size)
    return obj_mask


def nucleus_watershed_segmentation(obj_mask,min_dist):
    distance = ndi.distance_transform_edt(obj_mask)
    local_max_coords = ski.feature.peak_local_max(
        distance, min_distance=int(np.ceil(min_dist)), exclude_border=False
    )
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = ski.measure.label(local_max_mask)

    segmented_obj = ski.segmentation.watershed(-distance, markers, mask=obj_mask)
    return segmented_obj

def nuclear_seed_detection(nuclear_mask,nuclear_label,min_dist):
    distance = ndi.distance_transform_edt(nuclear_mask)
    local_max_coords = ski.feature.peak_local_max(
        distance,
        min_distance = int(np.ceil(min_dist)),
        labels = nuclear_label
    )
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    seeds = ski.measure.label(local_max_mask)

    return seeds

def produce_otsu_tresh(img):
    tmp_quant = np.quantile(img,[0.25,0.75])
    tmp_IQR = tmp_quant[1] - tmp_quant[0]
    tmp_n = img.shape[0] * img.shape[1]
    optim_width = 2 * tmp_IQR * tmp_n**(-1/3)
    tmp_nbin = ((np.max(img) - np.min(img))/optim_width).astype(np.uint)
    tmp_thresh = ski.filters.threshold_otsu(img,nbins=tmp_nbin)
    return tmp_thresh

def create_cytoplasm_watershed_forced_landscape(img,nuclear_mask,otsu_tresh):
    selem = ski.morphology.disk(1)
    nuclear_suppression_mask = ski.morphology.binary_dilation(nuclear_mask, selem)
    low_otsu = 0.5 * otsu_tresh
    high_otsu = otsu_tresh
    canny_edges = ski.feature.canny(
        img, 
        sigma=1.0, 
        low_threshold=low_otsu, # Relative to max intensity
        high_threshold=high_otsu
    )

    modified_canny_map = np.copy(canny_edges)
    modified_canny_map[nuclear_suppression_mask] = False 

    distance_to_canny_background = ndi.distance_transform_edt(~modified_canny_map) # ~canny_edges means non-edges are foreground
    cytoplasm_landscape = -distance_to_canny_background
    absolute_min_landscape_value = np.min(cytoplasm_landscape)
    forced_landscape = np.copy(cytoplasm_landscape)
    forced_landscape[nuclear_suppression_mask] = absolute_min_landscape_value

    return forced_landscape

def generate_nucleus_seeded_cyto_mask(img,nuclear_mask,nuclear_seeds):

    otsu_tresh = produce_otsu_tresh(img)

    forced_landscape = create_cytoplasm_watershed_forced_landscape(img,nuclear_mask,otsu_tresh)
    otsu_thresh_img = img > (otsu_tresh*0.5)
    cyto_mask = ski.morphology.remove_small_objects(otsu_thresh_img, min_size=200 *0.5)

    segmented_cyto = ski.segmentation.watershed(forced_landscape, markers = nuclear_seeds,mask = cyto_mask)

    return segmented_cyto


def detect_cell_clumps(segmented_cyto,nuclear_labels):
    cyto_size = np.unique_counts(segmented_cyto) 
    cell_props_table = ski.measure.regionprops_table(nuclear_labels, properties=('label', 'coords'))
    dfs = []
    for nucleus_id, coords in zip(cell_props_table['label'], cell_props_table['coords']):
        rows = coords[:, 0]
        cols = coords[:, 1]
        cytoplasm_id_mode = ndi.label(nuclear_labels == nucleus_id)[0] # Get current nucleus mask
        overlapping_cytoplasm_ids = segmented_cyto[rows, cols]
        valid_ids = overlapping_cytoplasm_ids[overlapping_cytoplasm_ids != 0]
        valid_id_count = np.unique_counts(valid_ids)

        tmp_pixel_count_df = pd.DataFrame({'cyto_ID':valid_id_count[0],'cyto_over_pixel_count':valid_id_count[1]})
        dfs.append(tmp_pixel_count_df.assign(nuc_ID = nucleus_id,nuc_pixel_count = coords.shape[0]))

    nuc_id_summary = pd.concat(dfs)
    nuc_id_summary = nuc_id_summary.merge(pd.DataFrame({'cyto_ID':cyto_size[0],'cyto_pixel_count':cyto_size[1]}))
    nuc_to_cyto_edgelist_df = (nuc_id_summary.assign(nuc_node = lambda df: ['nuc_' + str(idx) for idx in df.nuc_ID] ,
                        cyto_node = lambda df: ['cyto_' + str(idx) for idx in df.cyto_ID])
                .loc[:,['nuc_node','cyto_node']]
                        )

    B = nx.Graph()

    # Add nodes from the first partition (U) and set attribute 'bipartite' to 0
    B.add_nodes_from(nuc_to_cyto_edgelist_df.nuc_node.drop_duplicates().to_list(), bipartite=0)

    # Add nodes from the second partition (V) and set attribute 'bipartite' to 1
    B.add_nodes_from(nuc_to_cyto_edgelist_df.cyto_node.drop_duplicates().to_list(), bipartite=1)
    B.add_edges_from(nuc_to_cyto_edgelist_df.to_numpy())
    components = list(nx.connected_components(B))
    components_df = (pd.DataFrame({'cmpnt_size':[len(x) for x in components],'cmpnt_members':components}))

    cell_clump_df = components_df.query('cmpnt_size >2').reset_index(drop=True)

    return cell_clump_df


def correct_cell_clumps(segmented_cyto,cell_clump_df,nuclear_labels,forced_landscape):
    corrected_cytoplasm_labels = np.copy(segmented_cyto)
    cyto_props = ski.measure.regionprops_table(segmented_cyto, properties=('label', 'coords'))

    for idx in range(cell_clump_df.shape[0]):
        tmp_cmpnt = cell_clump_df.cmpnt_members.iloc[idx]
        segementation_mask_id_df =pd.concat([pd.DataFrame({'ID':[str(i).split('_')[1]],
                         'type':[str(i).split('_')[0]]}) 
                         for i in tmp_cmpnt])


        cyto_id = segementation_mask_id_df.query('type == "cyto"').ID.astype(int).to_list()
        nuc_id = segementation_mask_id_df.query('type == "nuc"').ID.astype(int).to_list()

        tmp_cyto_df = pd.DataFrame(cyto_props).query('label in @cyto_id')

        # Check for the splitting condition
        # --- A. Define Local Seeds and Mask ---
        # Mask of the current problematic cytoplasm region
        cyto_mask_merged =  np.isin(segmented_cyto,cyto_id)

        current_seeds = np.zeros_like(nuclear_labels)
        for n_id in nuc_id:
            current_seeds[nuclear_labels == n_id] = n_id
        landscape_split = np.copy(forced_landscape)
        landscape_split[~cyto_mask_merged] = 1.0 

        split_labels = ski.segmentation.watershed(
            image=landscape_split,
            markers=current_seeds,
            mask=cyto_mask_merged
        )

        # Further filter for cell labels overlapping single nucleus labels
        # What is the correct logic:
        #   1. given the split label we only find a single nucleus label within
        #   2. given a nuclear label we only find a single split label within   
        ok_cell_segmentation_id = []
        split_label_coord_dict = ski.measure.regionprops_table(split_labels, properties=('label', 'coords'))
        for nucleus_id, coords in zip(split_label_coord_dict['label'], split_label_coord_dict['coords']):
            rows = coords[:, 0]
            cols = coords[:, 1]
            overlapping_nuclei_ids = nuclear_labels[rows, cols]
            valid_ids = overlapping_nuclei_ids[overlapping_nuclei_ids != 0]
            if len(valid_ids) > 0:
                ok_cell_segmentation_id.extend(np.unique(valid_ids).tolist())
        
        update_mask = np.isin(split_labels,ok_cell_segmentation_id) & cyto_mask_merged
        corrected_cytoplasm_labels[update_mask] = split_labels[update_mask]

    return corrected_cytoplasm_labels

def correct_cell_segmentation_for_cell_clumping(img,segmented_cyto,nuclear_mask,nuclear_labels):
    otsu_tresh = produce_otsu_tresh(img)
    cell_clump_df = detect_cell_clumps(segmented_cyto,nuclear_labels)
    forced_landscape = create_cytoplasm_watershed_forced_landscape(img,nuclear_mask,otsu_tresh)
    corrected_cytoplasm_segmentation =  correct_cell_clumps(segmented_cyto,cell_clump_df,nuclear_labels,forced_landscape)
    tmp_clump_df = detect_cell_clumps(corrected_cytoplasm_segmentation,nuclear_labels)
    correct_segmentation = corrected_cytoplasm_segmentation.copy()
    while(tmp_clump_df.shape[0] > 0):
        correct_segmentation = correct_cell_segmentation_for_cell_clumping(img,correct_segmentation,nuclear_mask,nuclear_labels)
        tmp_clump_df = detect_cell_clumps(correct_segmentation,nuclear_labels)

    return correct_segmentation




