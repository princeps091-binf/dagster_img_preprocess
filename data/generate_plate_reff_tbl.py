#%%
import os
import pandas as pd
#%%
img_files = os.listdir('./raw/')
#%%
file_tbl = (pd.DataFrame({'file':img_files})
 .assign(plate = lambda df: [f.split('_')[0] for f in df.file.to_list()],
         well = lambda df: [f.split('_')[1] for f in df.file.to_list()],
         fov_channel = lambda df: [f.split('_')[2] for f in df.file.to_list()],
         fov = lambda df: [f[5:9] for f in df.fov_channel.to_list()],
         channel = lambda df: [f[-7:-4] for f in df.fov_channel.to_list()],

         )
 .loc[:,['plate','well','fov','channel','file']]
 .assign(fov_id = lambda df: df.plate
                            .add('_')
                            .add(df.well)
                            .add('_')
                            .add(df.fov))
)
file_tbl.to_csv('./../config/plate_mapping.csv',index=False)
# %%
