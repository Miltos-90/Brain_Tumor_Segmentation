
# Preprocessing parameters
PATCH_SIZE           = (128, 128)        # patch size
PATCH_STRIDE         = (64, 64)          # patch overlap
CLASS_RATIO          = 0.05              # Percentage of every slice containing at least one tumor class
SEED                 = 123               # seed for train/val/test split
TRAIN_RATIO          = 0.7               # Ratio of dataset to keep for the train set
VAL_RATIO            = 0.15              # Ratio of dataset to keep for the validation set
LOADER_WORKERS       = 0                 # No. workers for the dataloaders

# Learning parameters
DEVICE               = 'cuda'
EPOCHS               = 100               # Total training epochs
BATCHSIZE            = 128               # Train / validation loader batchsize
ENCODER_NAME         = 'resnet18'        # Backbone. See: https://smp.readthedocs.io/en/latest/encoders.html
ENCODER_WEIGHTS      = None              # Pretrained weights
ENCODER_DEPTH        = 3                 # No. downsampling operations in the encoder
ENCODER_IN_CHANNEL   = 3                 # Input channels for the model
DECODER_CHANNELS     = (64, 32, 16)      # list of numbers of Conv2D layer filters in decoder blocks
NO_CLASSES           = 4                 # No. classes in the segmentation task
LOSS_GAMMA           = 2                 # Focal loss exponent
SCHEDULER_FACTOR     = 0.33              # Multiplicative factor to reduce learning rate on plateau
SCHEDULER_PATIENCE   = 2                 # Number of epochs with no improvement after which learning rate will be reduced
LEARN_RATE           = 1e-4              # Optimizer learning rate
WEIGHT_DECAY         = 1e-4

# Path containing the raw MRI images
RAW_DATA_PATH        = './data/MICCAI_BraTS2020_TrainingData/'

# Paths containing the train/validation processed (sliced and patched) data
TRAIN_FILE           = './data/brats2020-processed/train.h5'
VAL_FILE             = './data/brats2020-processed/val.h5'

# Paths for checkpoints and logs
LOG_FILE             = 'logfile.txt'
LAST_CHECKPOINT      = 'checkpoint.pt'
BEST_CHECKPOINT      = 'best_checkpoint.pt'
