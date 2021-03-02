import bdpy
from bdpy.mri.image import export_brain_image
import numpy as np

data = bdpy.BData('kamitani_data/fmri/Subject3.mat')
data.show_metadata() # Prints the metadata out to the console

voxel_data = data.select('ROI_VC') # Returns the voxel data with shape (sample, num_voxels)
datatype = data.get_metadata('DataType')
print(datatype)


# bdpy function could be improved to not require a template when given xyz coords,
x = data.get_metadata('voxel_x', where='VoxelData')
y = data.get_metadata('voxel_y', where='VoxelData')
z = data.get_metadata('voxel_z', where='VoxelData')
xyz = np.vstack((x, y, z))
