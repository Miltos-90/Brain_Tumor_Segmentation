
from patcher import Patcher
import config
import numpy as np
import nibabel as nib
import torch
import h5py
from tqdm.notebook import tqdm

''' Preprocesses the raw data to generate the patches that will be 
    used for further training
'''
def makeDataset(dataframe, filename,
                patchSize       = config.PATCH_SIZE, 
                patchStide      = config.PATCH_STRIDE,
                classPatchRatio = config.CLASS_RATIO):
    
    patcher   = Patcher(kernel = patchSize, stride = patchStide)  
    iterable  = tqdm(dataframe.iterrows(), total = dataframe.shape[0])
    noPatches = 0
    
    with h5py.File(filename, 'a') as hf:
        
        for idx, paths in iterable:
            
            paths.drop('t1', axis = 0, inplace = True) # Remove t1 file    
            try:    
                x, y = loadTensors(paths.values) # Some folders are missing their segmentations
            except: 
                print(f'Failed on row {idx}.')
                
            else:
                x = torch.permute(x, (3, 0, 1, 2)) # Realign dimensions (C x H x W x D -> D (=B) x C x H x W)
                y = torch.permute(y, (3, 0, 1, 2)) # Depth dimension is considered as the batch dimension. One slice is an input image to be segmented
                x, y  = patcher.transform(x), patcher.transform(y)
                x, y  = discardPatches(x, y, classPatchRatio)
                noPatches = writePatches(x, y, hf, noPatches)

        print(f'{noPatches} patches written.')

    return 


''' Loads tensors from one datapoint '''
def loadTensors(paths): # in preprocess.py
    
    x = [] # Holds input tensors read from the data
    for path in paths:

        # Read required image from disk (uncached). Dimensions (H x W x D)
        data = np.asarray(nib.load(path).dataobj)

        if 'seg' in path:
            y = data.astype(np.int32)
            y = torch.tensor(y).long()
            y = y[None, ...] # Add channel dimension ( H x W x D -> C x H x W x D)
        else:
            tensor = torch.tensor(data).float()
            x.append(tensor)

    # Concatenate along the channel dimension Dimensions (C x H x W x D)
    x = torch.stack(x, dim = 0)

    return x, y


''' Discards MRI slices containing mostly background'''
def discardPatches(x, y, classCoverageRatio = config.CLASS_RATIO):

    # No. pixels in every patch
    totalPatchPixels = y.shape[2] * y.shape[3]

    # No. pixels on every patch containing at least one class
    tumorPatchPixels = (y[:, ...] != 0).sum(dim = (1, 2, 3) )

    # Get slices containing at least one class
    patchesToKeep = tumorPatchPixels >= totalPatchPixels * classCoverageRatio

    x = x[patchesToKeep, ...]
    y = y[patchesToKeep, ...]
    
    return x, y

''' Writes extracted patches to an h5 file with a unique index'''
def writePatches(x, y, hfile, noWrittenPatches):
    
    # Concatenate along the channel dimension (last channel is the segmentation map)
    data = torch.cat([x, y], dim = 1)
    for i in range(x.shape[0]):
        hfile.create_dataset(str(noWrittenPatches), data = data[i, ...])
        noWrittenPatches += 1

    return noWrittenPatches