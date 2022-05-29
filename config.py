FILEPATH             = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
DEVICE               = 'cuda'
LOG_FILE             = 'logfile.txt'
LAST_CHECKPOINT      = 'checkpoint.pt'
BEST_CHECKPOINT      = 'best_checkpoint.pt'
NO_CLASSES           = 4                 # No. classes in the segmentation task
LOADER_WORKERS       = 0                 # No. workers for the dataloaders
TEST_RATIO           = 0.15              # Ratio of dataset to keep for the test set
VAL_RATIO            = 0.15              # Ratio of dataset to keep for the validation set
BACKGROUND_RATIO     = 0.1               # Ratio of background-only patches to retain during training (% of total)
TRANSFORM_TYPE       = 'full'            # / 'minimal'. Type of image augmentation to be performed. See engine.py
ENCODER_NAME         = 'efficientnet-b0' # Unet backbone model. See: https://smp.readthedocs.io/en/latest/encoders.html
ENCODER_WEIGHTS      = None              # Pretrained weights
ENCODER_IN_CHANNEL   = 2                 # Input channels for the model
LOSS_ALPHA           = 2.0/100.0         # Focal loss alpha parameter
LOSS_GAMMA           = 2.4               # Focal loss gamma parameter
LOSS_NORM            = False             # Compute normalised version of focal loss
SCHEDULER_FACTOR     = 0.33              # Multiplicative factor to reduce learning rate on plateau
SCHEDULER_PATIENCE   = 4                 # Number of epochs with no improvement after which learning rate will be reduced
BATCHSIZE            = 192               # Train / validation / test loader batchsize
LEARN_RATE           = 2.48e-3           # Optimizer learning rate
EPOCHS               = 40                # Total training epochs
IMG_SIZE             = (160, 160, 1)     # Image dimensions at the end of augmentations
LABEL_REMAP = {0:0, 1:1, 2:2, 4:3}       # Fix messed-up labeling on the dataset

# Install additional packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torchio", "--quiet"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "git+https://github.com/qubvel/segmentation_models.pytorch", "--quiet"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-lr-finder", "--quiet"])