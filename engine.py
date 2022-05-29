# Libraries
import config
import torchio as tio 
import torch
from torch import nn
from torch.utils.data import random_split
import numpy as np
from torch.cuda import amp
from tqdm.notebook import tqdm

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


''' Predict on an independent test set'''
def predict(model, dataloader, batchSize, noSamples, noClasses = config.NO_CLASSES):

    # Matrices to hold statistics
    size = (noSamples, noClasses)
    TP   = torch.empty(size = size)
    TN   = torch.empty(size = size)
    FP   = torch.empty(size = size)
    FN   = torch.empty(size = size)

    model.eval()
    with torch.no_grad():
        for batchNo, (x, target) in tqdm(enumerate(dataloader), 
                                         total = int(noSamples / batchSize)):
            # Predict
            yhat = model(x)
            pred = torch.argmax(yhat, dim = 1).long()

            # Compute stats
            tp, fp, fn, tn = smp.metrics.get_stats(pred, target.long(), 
                                                   mode = 'multiclass', 
                                                   num_classes = noClasses)

            # Get indices to fill for the matrices holding results on the entire test set
            startIdx = batchNo * D
            endIdx   = (batchNo + 1) * D

            # Add batch results
            TP[startIdx:endIdx, :] = tp
            FP[startIdx:endIdx, :] = fp
            TN[startIdx:endIdx, :] = tn
            FN[startIdx:endIdx, :] = fn

    return TP, FP, FN, TN


''' Augmentations for the train and validation/test sets'''
def makeTransforms(flairPaths, t2Paths,
                   transformationType, # full / minimal
                   targetSize   = config.IMG_SIZE, 
                   labelMapping = {0:0, 1:1, 2:2, 4:3}, 
                   noClasses    = config.NO_CLASSES, 
                   include      = ['flair_mri', 't2_mri', 'mask'],
                   label_keys   = 'mask'):
    
    # Make trainable landmarks
    landmarks  = _trainLandmarks(flairPaths, t2Paths)
    
    # Make full training transformation
    fullTransform = tio.Compose([                    # -------- Normalisation -------  
        tio.HistogramStandardization(             # standard histogram of the foreground
            landmarks,
            include    = include,
            label_keys = label_keys
        ),                             
        tio.ZNormalization(                       # zero mean, unit variance of foreground
            masking_method = _maskingFunction,
            include        = include,
            label_keys     = label_keys
        ),                                        # ------- Spatial transformations ------
        tio.OneOf({                               # either
            tio.RandomFlip(                           # Flip images
                axes             = (0, 1),
                flip_probability = 0.5,
                include          = include,
                label_keys       = label_keys
            ) : 0.8,
            tio.RandomAffine(                         # simulate different positions and size of the patient
                include    = include,
                label_keys = label_keys
            ) : 0.8,                                
            tio.RandomAnisotropy(                     # simulate down- and then up-sampling
                downsampling = (1.5, 2.5),
                axes         = (0, 1),
                include      = include,
                label_keys   = label_keys
            ) : 0.3,
            tio.RandomElasticDeformation(         # interpolate displacement fields using cubic B-splines.
                max_displacement = (2, 2, 0),
                include          = include,
                label_keys       = label_keys 
            ) : 0.3}, 
            p          = 0.9,                     # to 90% of images
            include    = include,
            label_keys = label_keys
        ),                                        # -------- Augmentations --------
        tio.OneOf({                               # either
            tio.RandomNoise(                          # Gaussian noise 25% of times
                p          = 0.5,
                std        = 0.15,
                include    = include,
                label_keys = label_keys
            ),
            tio.RandomBlur(                           # Smooth / blur image 25% of times
                p          = 0.5,
                std        = (0, 0.2),
                include    = include,
                label_keys = label_keys
            ),                               
            tio.RandomBiasField(                      # magnetic field inhomogeneity
                coefficients = 0.1, 
                p            = 0.30,
                include      = include,
                label_keys   = label_keys
            )},
            p          = 0.9,                     # to 90% of images
            include    = include,
            label_keys = label_keys
        ),
        tio.CropOrPad(                            # Tight crop around the brain
            targetSize,
            include    = include,
            label_keys = label_keys
        ),                  
        tio.RemapLabels(                          # Remap labels
            labelMapping,
            include    = include,
            label_keys = label_keys
        ),
        tio.Mask(                                 # Set values outside mask to constant value
            masking_method = _maskingFunction,
            include        = include,
            label_keys     = label_keys
        )
    ], 
    copy       = False,
    include    = include,
    label_keys = label_keys)
    
    minTransform = tio.Compose([
        tio.HistogramStandardization(
                landmarks,
                include    = include,
                label_keys = label_keys
        ),     
        tio.ZNormalization(
            masking_method = _maskingFunction,
            include        = include,
            label_keys     = label_keys
        ),
        tio.CropOrPad(
            targetSize,
            include     = include,
            label_keys  = label_keys
        ),
        tio.RemapLabels(
            labelMapping,
            include     = include,
            label_keys  = label_keys
        ),
        tio.Mask(
            masking_method = _maskingFunction,
            include        = include,
            label_keys     = label_keys
        )],
    copy       = False,
    include    = include,
    label_keys = label_keys)
    
    if transformationType == 'full':      trainTransform = fullTransform 
    elif transformationType == 'minimal': trainTransform = minTransform 

    valTransform = minTransform

    return trainTransform, valTransform

''' Trains the necessary landmarks for image augmentation'''
def _trainLandmarks(flairPaths, t2Paths):
    landmarks  = {'flair_mri': tio.HistogramStandardization.train(
                                    images_paths     = flairPaths,
                                    masking_function = _maskingFunction),
                  't2_mri':    tio.HistogramStandardization.train(
                                    images_paths     = t2Paths, 
                                    masking_function = _maskingFunction)
                 }
    return landmarks
    

''' Function that returns a bool tensor indicating the mask-related pixels of an input MRI.
    Used in the Transformations.
'''
def _maskingFunction(tensor):
    mask = torch.zeros_like(tensor, dtype = bool)
    mask[tensor > 0] = True
    
    return mask