# Libraries
import config
import os
import torchio as tio
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import nibabel as nib
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

''' Dataset class '''
class BRATSDataset(Dataset):
    
    def __init__(self, flairImageDir, t2ImageDir, segmentationDir, patchNumber, transform):
        
        self.flairImages = flairImageDir
        self.t2Images    = t2ImageDir
        self.labels      = segmentationDir
        self.patch       = patchNumber
        self.transform   = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        # Get filenames
        flairPath = self.flairImages[idx]
        t2Path    = self.t2Images[idx]
        segPath   = self.labels[idx]
        patchNo   = self.patch[idx]
        
        # Read from disk (uncached). Dimensions (Width, Height)
        flair = np.asarray(nib.load(flairPath).dataobj)[:, :, patchNo]
        t2    = np.asarray(nib.load(t2Path).dataobj)[:, :, patchNo]
        y     = np.asarray(nib.load(segPath).dataobj)[:, :, patchNo].astype(np.uint8)

        # Expand dimensions [C x H x W x D] (required for the transformations)
        flair = flair[np.newaxis, :, :, np.newaxis]
        t2    = t2[np.newaxis, :, :, np.newaxis]
        y     = y[np.newaxis, :, :, np.newaxis]

        # Apply transformations
        out  = self.transform({'flair_mri': torch.tensor(flair),
                               't2_mri':    torch.tensor(t2), 
                               'mask':      torch.tensor(y)})

        x = torch.cat((out['flair_mri'], out['t2_mri']), dim = 0)
        y = out['mask']
        
        # Remove depth dimension
        x = x.squeeze(dim = -1)
        y = y.squeeze(dim = -1)
        
        # Remove channel dimension from targets
        y = y.squeeze(dim = 0)
        
        return x, y

''' Convenience function to return a dataloader '''
def makeDataloader(df, transform, shuffle,
                   batchSize = config.BATCHSIZE, 
                   workers   = config.LOADER_WORKERS):
    
    # Make a dataset
    dataset = BRATSDataset(flairImageDir   = df['flairPath'].values,
                           t2ImageDir      = df['t2Path'].values,
                           segmentationDir = df['labelPath'].values,
                           patchNumber     = df['patch'].values,
                           transform       = transform)
    
    # Make dataloader
    dataloader = DataLoader(
        dataset            = dataset,
        batch_size         = batchSize,
        num_workers        = workers,
        shuffle            = shuffle,
        pin_memory         = True,
        persistent_workers = False)
    
    return dataloader


''' Resamples the df to downsample the background-only patches, and remove empty patches''' 
def subsample(df, backgroundRatio, seedNo = 45): # fraction of patches to keep containing only background label
    
    np.random.seed(seedNo) # Reproducibility

    # List containing patches with foreground labels
    foregroundLabel = df.index[df['containsNCRNET'] | df['containsED'] | df['containsET']].values

    # List containing patches with background label only
    backgroundOnly  = df.index[df['BKGPatch']].values

    # Subsample the background-only patches
    noBackgroundPatches = len(backgroundOnly)

    # Randomly select a number of patches to keep from the background-only patches
    backgroundOnlyIdx = np.random.randint(low = 0, high = noBackgroundPatches, size = int(noBackgroundPatches * backgroundRatio))
    backgroundOnly    = backgroundOnly[backgroundOnlyIdx]

    # Merge the subsampled backgroung patches w/ the foreground patches
    samplesToKeep = np.union1d(foregroundLabel, backgroundOnly)

    return df.loc[samplesToKeep, :]


''' Split data into two disjoint subsets based on a group column'''
def GroupTrainTestSplit(groupColumn, testRatio, seedNo = 45):
    
    # Shuffle
    groupColumn = groupColumn.sample(frac = 1, random_state = seedNo)
    
    np.random.seed(seedNo) # Reproducibility

    # Get groups to appear in train and test sets
    noUniqueGroups = groupColumn.unique().shape[0]
    testGroups     = np.random.randint(low = 0, high = noUniqueGroups - 1, size=int(noUniqueGroups * testRatio))
    trainGroups    = np.setdiff1d(groupColumn, testGroups)

    # Get indices of the corresponding groups
    trainGroupIdx = groupColumn[groupColumn.isin(trainGroups)].index.tolist()
    testGroupIdx  = groupColumn[groupColumn.isin(testGroups)].index.tolist()

    return trainGroupIdx, testGroupIdx


''' Make a generic pytorch checkpoint '''
def makeCheckpoint(epoch, model, optimizer, scheduler, scaler, trainLoss, valLoss, bestValLoss, verbose = True):
    
    if verbose:
        print(f'Epoch {epoch} | train loss: {np.round(trainLoss, 5)} | val loss: {np.round(valLoss, 5)}')
    
    # Check if improvement was found
    if valLoss < bestValLoss:
        if verbose: print('Found new best.')
        bestValLoss = valLoss
        filePath    = config.BEST_CHECKPOINT
    else:
        filePath = config.LAST_CHECKPOINT
        
    # Save to the appropriate file
    if verbose: print('Making checkpoint...')
        
    torch.save({'epoch'                : epoch,
                'model_state_dict'     : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'scaler_state_dict'    : scaler.state_dict(),
                'train_loss'           : trainLoss,
                'validation_loss'      : valLoss,
                'best_validation_loss' : bestValLoss,
                }, filePath)
    
    return bestValLoss


''' Load a genetic pytorch checkpoint'''
def loadCheckpoint(model, optimizer, scheduler, scaler, filePath):
    
    if os.path.exists(filePath):

        print('Loading checkpoint...')
        
        checkpoint = torch.load(filePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch       = checkpoint['epoch'] + 1 # Get epoch to be run now
        trainLoss   = checkpoint['train_loss']
        valLoss     = checkpoint['validation_loss']
        bestValLoss = checkpoint['best_validation_loss']

    else:
        
        print('No checkpoint found.')

        epoch       = 0
        trainLoss   = None
        valLoss     = None
        bestValLoss = np.inf
    
    return model, optimizer, scheduler, scaler, epoch, trainLoss, valLoss, bestValLoss


''' Make lists containing paths to the input files and corresponding masks '''
def _zipDirectories(filepath = config.FILEPATH):
    
    flairFiles, t1Files, t1ceFiles, t2Files, outputFiles  = [], [], [], [], []
    
    # Loop over all folders (each folder contains training sample)
    for folder in os.listdir(filepath):
        subfolder = os.path.join(filepath, folder)
        
        if os.path.isdir(subfolder):
            fileList  = os.listdir(subfolder)
            try:
                t1File    = np.argmax(['_t1.nii'    in file for file in fileList])
                t1ceFile  = np.argmax(['_t1ce.nii'  in file for file in fileList])
                t2File    = np.argmax(['_t2.nii'    in file for file in fileList])
                flairFile = np.argmax(['_flair.nii' in file for file in fileList])
                outFile   = np.argmax(['_seg.nii'   in file for file in fileList])
                                
                flairFiles.append(  os.path.join(subfolder, fileList[flairFile]) )
                t1Files.append(     os.path.join(subfolder, fileList[t1File])    )
                t1ceFiles.append(   os.path.join(subfolder, fileList[t1ceFile])  )
                t2Files.append(     os.path.join(subfolder, fileList[t2File])    )
                outputFiles.append( os.path.join(subfolder, fileList[outFile])   )
            except: # File 355 does not contain a valid segmentation file
                pass
    
    return zip(flairFiles, t1Files, t1ceFiles, t2Files, outputFiles)


''' Collects metadata from the entire dataset '''
def collectMetadata(filepath = config.FILEPATH):
    
    df = [] # List of dataframes containing metadata
    noFiles = 369

    for subjectNo, (flairPth, t1Pth, t1cePth, t2Pth, labPth) \
    in tqdm(enumerate(_zipDirectories(filepath)), total = noFiles):

        x = nib.load(flairPth).get_fdata()
        y = nib.load(labPth).get_fdata()

        H, W, D = x.shape
        HW = H * W

        xFlat = x.reshape(1, -1, D).squeeze() # Flatten height, width dimension
        yFlat = y.reshape(1, -1, D).squeeze() # Flatten height, width dimension

        emptyPatch         = np.all(xFlat == 0, axis = 0) # Completely empty patch
        BKGpixelPercent    = np.sum(yFlat == 0, axis = 0) / HW * 100
        NCRNETpixelPercent = np.sum(yFlat == 1, axis = 0) / HW * 100
        EDpixelPercent     = np.sum(yFlat == 2, axis = 0) / HW * 100
        ETpixelPercent     = np.sum(yFlat == 4, axis = 0) / HW * 100

        containsNCRNET  = np.any(yFlat == 1, axis = 0) # Patch with label 1
        containsED      = np.any(yFlat == 2, axis = 0) # Patch with label 2
        containsET      = np.any(yFlat == 4, axis = 0) # Patch with label 4
        containsBKGonly = (BKGpixelPercent > 0) & ~(emptyPatch | containsNCRNET | containsED | containsED)
        patchNo         = np.arange(D)                 # Patch ID

        # Make dataframe
        curdf = pd.DataFrame.from_dict({
        'patch':          patchNo,
        'emptyPatch':     emptyPatch,
        'BKGPatch':       containsBKGonly,
        'containsNCRNET': containsNCRNET,
        'containsED':     containsED,
        'containsET':     containsET,
        'NCRNETpixels':   NCRNETpixelPercent,
        'EDpixels':       EDpixelPercent,
        'ETpixels':       ETpixelPercent,
        'BKGpixels':      BKGpixelPercent,
        })

        curdf['flairPath'] = flairPth
        curdf['t1Path']    = t1Pth
        curdf['t1cePath']  = t1cePth
        curdf['t2Path']    = t2Pth
        curdf['labelPath'] = labPth
        curdf['subjectID'] = subjectNo

        # Append to list
        df.append(curdf)

    # concatenate
    df = pd.concat(df, axis = 0, ignore_index = True)
    
    return df


''' Prints a summary of the metadata collected by the collectMetadata() function '''
def makeSummary(df):

    emptyPatches   = df['emptyPatch'].sum()
    NCR_NETpatches = df['containsNCRNET'].sum()
    EDpatches      = df['containsED'].sum()
    ETpatches      = df['containsET'].sum()
    bkgOnlyPatches = df['BKGPatch'].sum()
    allLabels      = (df['containsNCRNET'] & df['containsED'] & df['containsET']).sum()
    noSubjects     = df['subjectID'].unique().shape[0]

    print(f'No. subjects: {noSubjects}')
    print(f'No. patches: {df.shape[0]}')
    print(f'No. empty patches: {emptyPatches}')
    print(f'No. patches containing only background: {bkgOnlyPatches}')
    print(f'No. patches containing NCR/NET label: {NCR_NETpatches}')
    print(f'No. patches containing ED label: {EDpatches}')
    print(f'No. patches containing ET label: {ETpatches}')
    print(f'No. patches containing all labels: {allLabels}')

    metaSummary = df[['BKGpixels', 'NCRNETpixels', 'EDpixels', 'ETpixels']].describe()
    
    return metaSummary.loc[['mean', 'std', 'max']]


''' Plots a batch from the dataloader'''
def plotBatch(x, y):
    
    B, C = x.shape[0], x.shape[1]

    for sampleNo in range(B): # Loop over batches

        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 5))

        for channel in range(C): # Loop over channels (image types)

            # Plot image type
            img = x.numpy()[sampleNo, channel, :, :]
            ax[channel].imshow(img, cmap = 'bone')
            ax[channel].axis('off');

        # Plot label
        labels = y[sampleNo, :, :, :].squeeze(dim = 0)
        ax[-1].imshow(labels)
        ax[-1].axis('off');
        
    return 
    
    
''' Evaluates several metrics on a given dataset '''
def computeMetrics(TP, FP, FN, TN, reduction = 'macro-imagewise'):

    accuracy    = smp.metrics.accuracy(    TP, FP, FN, TN, reduction = reduction).item()
    precision   = smp.metrics.precision(   TP, FP, FN, TN, reduction = reduction).item()
    recall      = smp.metrics.recall(      TP, FP, FN, TN, reduction = reduction).item()
    sensitivity = smp.metrics.sensitivity( TP, FP, FN, TN, reduction = reduction).item()
    specificity = smp.metrics.specificity( TP, FP, FN, TN, reduction = reduction).item()
    iou_score   = smp.metrics.iou_score(   TP, FP, FN, TN, reduction = reduction).item()
    f1_score    = smp.metrics.f1_score(    TP, FP, FN, TN, reduction = reduction).item()
    f2_score    = smp.metrics.fbeta_score( TP, FP, FN, TN, beta = 2, reduction = reduction).item()

    print(f'Image-wise macro-averaged Accuracy:\t {accuracy:.3f}')
    print(f'Image-wise macro-averaged Precision:\t {precision:.3f}')
    print(f'Image-wise macro-averaged Recall:\t {recall:.3f}')
    print(f'Image-wise macro-averaged Sensitivity:\t {sensitivity:.3f}')
    print(f'Image-wise macro-averaged Specificity:\t {specificity:.3f}')
    print(f'Image-wise macro-averaged F1 Score:\t {f1_score:.3f}')
    print(f'Image-wise macro-averaged F2 Score:\t {f2_score:.3f}')
    print(f'Image-wise macro-averaged IoU Score:\t {iou_score:.3f}')
    
    return
    
    
    