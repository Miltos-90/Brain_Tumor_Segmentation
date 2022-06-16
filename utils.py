

import config
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torchio as tio
from torch.utils.data import Dataset
import h5py

''' Dataset class'''
class BRATSDataset(Dataset):
    
    def __init__(self, filename, noSamples, transform):

        self.file  = h5py.File(filename, "r") # open h5 file
        self.noSamples = noSamples
        self.transform = transform

    def __len__(self):
        return self.noSamples

    def __getitem__(self, idx):

        # Read from file [Dimensions x: C x H x W, y: 1 x H x W]
        t = torch.tensor(self.file[str(idx)])    
        x = t[0:-1, :, :].float()
        y = t[-1, :, :].long()


        if self.transform is not None:
            # Expand dimensions [C x H x W x D] (required for the transformations)
            x, y = x[..., None], y[None, ..., None]
            
            # Apply transformations (split needed for proper normalisation of each channel)
            out  = self.transform({'x1': x[0,...][None, ...], # Dims: 1 x H x W x D
                                   'x2': x[1,...][None, ...], # Dims: 1 x H x W x D
                                   'x3': x[2,...][None, ...], # Dims: 1 x H x W x D
                                   'y' : y})                  # Dims: 1 x H x W x D

            x = torch.cat([out['x1'], out['x2'], out['x3']], dim = 0) # Dims: C x H x W x D
            y = out['y']                                              # Dims: 1 x H x W x D

            # Remove depth dimension (Dims: C(x)/1(y) x H x W)
            x, y = x.squeeze(dim = -1), y.squeeze(dim = -1)

        # Remove channel dimension from targets (Dims: H, W)
        y = y.squeeze(dim = 0)

        return x, y


''' Makes a dataframe containing paths to the input files and corresponding masks '''
def makeFileList(filepath = config.RAW_DATA_PATH):
    
    records = []

    for folder in os.listdir(filepath):
        subfolder = os.path.join(filepath, folder)

        if os.path.isdir(subfolder):
            filelist = sorted(os.listdir(subfolder))
            records.append([os.path.join(subfolder, file) for file in filelist])

    df = pd.DataFrame.from_records(records, columns = ['flair', 'seg', 't1', 't1ce', 't2'])
    
    return df


''' Plots a datapoint '''
def plotSample(paths):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (15, 5))
    
    for idx, path in enumerate(paths):

        x = np.asarray(nib.load(path).dataobj)
        D = x.shape[2]
        ax[idx].imshow(x[:, :, D // 2], cmap = 'bone')
        
        mriType = path.split('/')[-1].split('_')[-1].split('.')[0]
        ax[idx].set_title(mriType)
        ax[idx].axis('off')
        
    return 


''' Splits data in train/val/test sets'''
def split(df, 
          trainRatio = config.TRAIN_RATIO, 
          valRatio   = config.VAL_RATIO, 
          seed       = config.SEED):
    
    # Reproducibility
    np.random.seed(seed)

    # No samples in train/val/test sets
    samples      = df.shape[0]
    trainSamples = np.floor(samples * trainRatio).astype(int)
    valSamples   = np.floor(samples * valRatio).astype(int)
    testSamples  = samples - trainSamples - valSamples

    # Shuffle indices
    idx = np.arange(0, samples)
    np.random.shuffle(idx)

    # Choose indices for train/val/test sets
    trainidx = idx[0:trainSamples]
    validx   = idx[(trainSamples + 1): (trainSamples + valSamples)]
    testidx  = idx[(trainSamples + valSamples + 1) : samples]

    # Split datasets
    traindf = df.iloc[trainidx, :]
    valdf   = df.iloc[validx, :]
    testdf  = df.iloc[testidx, :]

    return traindf, valdf, testdf

''' Mask function for the transformations '''
def maskFunction(tensor):
    mask = torch.zeros_like(tensor, dtype = bool)
    mask[tensor > 0] = True
    return mask


# Make a generic pytorch checkpoint
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


# Load a genetic pytorch checkpoint
def loadCheckpoint(model, optimizer, scheduler, scaler, filePath, mapLocation = config.DEVICE):
    
    if os.path.exists(filePath):

        print('Loading checkpoint...')
        
        checkpoint = torch.load(filePath, map_location = mapLocation)
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


'''  Computes various metrics from TP, FP, FN, TN '''
def computeMetrics(TP, FP, FN , TN):

    ACC  = (TP + TN) / (TP + TN + FP + FN)
    P    = TP / (TP + FP)
    R    = TP / (TP + FN)
    F1   = 2 * P * R / (P + R)
    IOU  = TP / (TP + FP + FN)
    DICE = 2 * TP / (2 * TP + FP + FN)

    print('Precision:\t',  P[1:].cpu().numpy().round(3))
    print('Recall: \t',    R[1:].cpu().numpy().round(3))
    print('Accuracy:\t',   ACC[1:].cpu().numpy().round(3))
    print('F1 score:\t',   F1[1:].cpu().numpy().round(3))
    print('IoU score:\t',  IOU[1:].cpu().numpy().round(3))
    print('DICE score:\t', DICE[1:].cpu().numpy().round(3))
    
    return 


''' Save a sample to gif '''
def makeGifs(x, y, yhat, folder):

    os.makedirs(folder)

    mri   = tio.ScalarImage(tensor = torch.permute(x, (1,2,3,0)))
    tAct  = tio.LabelMap(tensor = torch.permute(y, (1,2,3,0)))
    tPred = tio.LabelMap(tensor = torch.permute(yhat, (1,2,3,0)))
    
    mri.to_gif(  output_path = folder + 'mri.gif', axis = 2, duration = 3, rescale = True)
    tAct.to_gif( output_path = folder + 'act.gif', axis = 2, duration = 3, rescale = True)
    tPred.to_gif(output_path = folder + 'pred.gif', axis = 2, duration = 3, rescale = True)
    
    return
