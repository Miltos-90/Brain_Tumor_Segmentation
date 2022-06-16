# Libraries
import config
import torch
import numpy as np
from torch.cuda import amp
from preprocess import loadTensors
from patcher import Patcher


''' Train function '''
def train(model, criterion, optimizer, scaler, loader, device = config.DEVICE, verbose = True):
    
    model.train()
    trainLoss = 0
    
    for batchNo, (x, y) in enumerate(loader):

        optimizer.zero_grad(set_to_none = True)
        x = x.to(device)
        y = y.to(device)
    
        with amp.autocast():
            yhat = model(x)
            loss = criterion(y_true = y, y_pred = yhat) 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        trainLoss += loss.item() * x.shape[0]

        if verbose: print(f'\t Training Batch {batchNo} loss: {np.round(loss.item(), 8)}')
        
    return trainLoss
    

''' Validation loss '''
def validate(model, criterion, loader, device = config.DEVICE, verbose = True):
    
    model.eval()
    valLoss = 0

    with torch.no_grad():
        for batchNo, (x, y) in enumerate(loader):

            x    = x.to(device)
            y    = y.to(device)
            yhat = model(x)
            loss = criterion(y_true = y, y_pred = yhat) 
            valLoss += loss.item() * x.shape[0]

            if verbose: print(f'\t Validation Batch {batchNo} loss: {np.round(loss.item(), 8)}')
    
    return valLoss


''' Returns predictions of one sample'''
def predict(paths, model, transform, 
            patchSize   = config.PATCH_SIZE, 
            patchStride = config.PATCH_STRIDE, 
            device      = config.DEVICE, 
            returnInput = True):
    
    # Objects to creae patches
    patcherX   = Patcher(kernel = patchSize, stride = patchStride)
    patcherY   = Patcher(kernel = patchSize, stride = patchStride)
    
    # load data from disk
    paths.drop('t1', axis = 0, inplace = True) # Remove t1 file
    x, y = loadTensors(paths.values)
    x    = x.to(device)
    y    = y.to(device)
        
    # Exract patches
    x    = torch.permute(x, (3, 0, 1, 2))               # Realign dimensions for patches (C x H x W x D -> D (=B) x C x H x W)
    y    = torch.permute(y, (3, 0, 1, 2))               # Depth dimension is considered as the batch dimension. One slice is an input image to be segmented
    x, y = patcherX.transform(x), patcherY.transform(y) # Extract patches (B x C x H x W)

    # Apply transformations
    if transform is not None:
        
        x = torch.permute(x, (1, 2, 3, 0)) 
        y = torch.permute(y, (1, 2, 3, 0))
    
        out = transform({'x1': x[0,...][None,...],     # Dims: 1 x H x W x D
                         'x2': x[1,...][None,...],     # Dims: 1 x H x W x D
                         'x3': x[2,...][None,...],     # Dims: 1 x H x W x D
                         'y' : y})                     # Dims: 1 x H x W x D

        x = torch.cat([out['x1'], out['x2'], out['x3']], dim = 0)  # Dims: C x H x W x D
        y = out['y']                                               # Dims: H x W x D
        
    # Predict
    x    = torch.permute(x, (3, 0, 1, 2))
    y    = torch.permute(y, (3, 0, 1, 2))
    yhat = model(x)
    yhat = torch.argmax(yhat, dim = 1, keepdim = True)
        
    # Convert patches to original dimensions
    patcherX.dtype, patcherY.dtype = torch.float, torch.float
    yhat = patcherY.inverse_transform(yhat).nan_to_num(0).long()
    y    = patcherY.inverse_transform(y).nan_to_num(0).long()
    x    = patcherX.inverse_transform(x).nan_to_num(-1)
    
    if returnInput: out = (x, y, yhat)
    else:           out = yhat.long()
        
    return out