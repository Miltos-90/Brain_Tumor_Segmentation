import torch
import torch.nn.functional as F

'''
Example call:
patcher = Patcher(kernel = (128, 128), stride = (64, 64))

p    = patcher.transform(x)           x: B x C x H x W -> p: patchBatch (=B * noPatchesH * noPatchesW) x C x kernel[0] x kernel[1]
xOut = patcher.inverse_transform(p)   p (as before) -> xOut: B x C x H x W (same as x)
'''

class Patcher(object):
    '''
        Initialisation function
        Inputs:
            kernel:  Kernel (i.e. patch) size (Height, Width)
            stride:  Stride (Height, Width)
    '''
    def __init__(self, kernel: (int, int), stride: (int, int)) -> None:
        '''
        Initialisation function
        Inputs:
            kernel:  Kernel (i.e. patch) size (Height, Width)
            stride:  Stride (Height, Width)
        '''
        
        self.kernel = kernel
        self.stride = stride
        
        # The following properties are inferred on the call to transform()
        # and are used on the call to inverse_transform().
        self.B, self.C, self.H, self.W = None, None, None, None
        self.pad   = [None, None] # padding vertical / horizontal direction (height, width)
        self.patch = [None, None] # No patches vertical / horizontal direction (height, width)
        self.dtype = None
    
    def transform(self, x:torch.tensor) -> torch.tensor:
        '''
        Converts an input tensor representating a batch of images to a tensor
        of patches, based on the kernel (patch) size and stride.
        The patches will or will not overlap based on the kernel size and stride.
        Inputs:
            x: torch.tensor w/ dimensions (Batch, Channel, Height, Width)
        Outputs:
            patch: torch.tensor w/ dimensions (Patch batch, Channel, kernelHeight, kernelWidth)
        '''
    
        self.B, self.C, self.H, self.W = x.shape
        self.dtype = x.dtype

        # Pad to multiples of kernel size
        padLR = self.W % self.kernel[1] // 2 # Left & right padding
        padTB = self.H % self.kernel[0] // 2 # Top & bottom padding
        
        # Dims: B, C, H, W
        xPadded = F.pad(x, (padLR, padLR, padTB, padTB))
        _, _, self.pad[0], self.pad[1] = xPadded.shape
        
        # Unfold. Dims: B, C, patchNoH, patchNoW, kernelH, kernelW
        patches = xPadded.unfold(2, self.kernel[0], self.stride[0])\
                         .unfold(3, self.kernel[1], self.stride[1])
        _, _, self.patch[0], self.patch[1], _, _ = patches.shape

        # Transform to Dims: B, patchNoH, patchNoW, C, kernelH, kernelW
        patches = patches.permute(0, 2, 3, 1, 4, 5) 
        
        # Transfrom to Dims: B x patchNoH x patchNoW, C, kernelH, kernelW
        patches = patches.reshape(-1, self.C, self.kernel[0], self.kernel[1])

        return patches.contiguous()
        
    def inverse_transform(self, patches:torch.tensor) -> torch.tensor:
        '''
            Converts an input tensor representing patches extracted from an image, to
            the image itself.
            Inputs:
                patches: torch.tensor w/ dimensions according to the transform output
                imgSize: Shape of the tensor to be constructed (Batch, Channel, Height, Width).
                         If none, the shape inferred during the call toPatches() will be used
            Outputs:
                x: torch.tensor w/ dimensions (Batch, Channel, Height, Width)
        '''
        
        # Dims: B, patchNoH, patchNoW, C, kernelH, kernelW
        patches = patches.view(self.B, self.patch[0], self.patch[1], self.C, self.kernel[0], self.kernel[1])
        
        # Dims: B, C, patchH, patchNoW, kernelH, kernelW
        patches = patches.permute(0, 3, 1, 2, 4, 5)
        
        # Dims: B, C, patchNoH x patchNoW, kernelH x kernelW
        patches = patches.view(self.B, self.C, -1, self.kernel[0] * self.kernel[1])
        
        # Dims: B, C, kernelH x kernelW, patchNoH x patchNoW
        patches = patches.permute(0, 1, 3, 2)
        
        # Dims: B, C x kernelH x kernelW, ...
        patches = patches.view(self.B, self.C * self.kernel[0] * self.kernel[1], -1)

        # Fold. Dims: B, C, Hpadded, Wpadded
        output = F.fold(patches.float(), output_size = self.pad, kernel_size = self.kernel, stride = self.stride)

        # Normalise overlaps
        eyeSz  = (self.B, self.C, self.pad[0], self.pad[1])
        I      = torch.ones(size = eyeSz).float()
        I      = F.unfold(I, kernel_size = self.kernel, stride = self.stride)
        nMap   = F.fold(I, output_size = self.pad, kernel_size = self.kernel, stride = self.stride)
        output /= nMap

        # Pad to original size. Dims: B, C, H, W
        padLR  = (self.pad[0] - self.H) // 2
        padTB  = (self.pad[1] - self.W) // 2
        output = F.pad(output, (-padLR, -padLR, -padTB, -padTB))
        
        return output.contiguous().type(self.dtype)