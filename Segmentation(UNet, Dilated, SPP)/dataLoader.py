import torch
import numpy as np
import os.path as osp
import random
from torch.utils.data import Dataset
from PIL import Image
import cv2

class BatchLoader(Dataset ):
    def __init__(self, imageRoot, labelRoot, fileList, imWidth = None, imHeight = None, numClasses=21, randomShuffle=True):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.labelRoot = labelRoot
        self.fileList = fileList

        with open(fileList, 'r') as fIn:
            imgNames = fIn.readlines()
        imgNames = [x.strip() for x in imgNames ]
        imgNames = sorted(imgNames )

        self.imgNames = [osp.join(imageRoot, x + '.jpg') for x in imgNames ]
        self.labelNames = [osp.join(labelRoot, x + '.png') for x in imgNames ]

        self.count = len(self.imgNames )
        self.perm = list(range(self.count ) )
        if randomShuffle:
            random.shuffle(self.perm )
        print('Image Num: %d' % self.count )

        # If image height and width are None
        # do not do any cropping
        self.imHeight = imHeight
        self.imWidth = imWidth

        # MEAN and std of image
        self.imMean = np.array([0.485, 0.456, 0.406], dtype=np.float32 )
        self.imStd = np.array([0.229, 0.224, 0.225], dtype = np.float32 )

        self.imMean = self.imMean.reshape([1, 1, 3] )
        self.imStd = self.imStd.reshape([1, 1, 3] )
        self.numClasses = numClasses

        self.iterCount = 0

    def __len__(self):
        return self.count

    def __getitem__(self, ind ):

        imName = self.imgNames[self.perm[ind] ]
        labelName = self.labelNames[self.perm[ind] ]

        im = self.loadImage(imName )
        label, labelIndex, mask = self.loadLabel(labelName )

        # If image size is given, randomly crop the images
        if not (self.imHeight is None or self.imWidth is None):
            nrows, ncols = im.shape[1], im.shape[2]
            gapH = (nrows - self.imHeight )
            gapW = (ncols - self.imWidth )
            rs = int(np.round(np.random.random() * gapH ) )
            cs = int(np.round(np.random.random() * gapW ) )

            im = im[:, rs:rs+self.imHeight, cs:cs+self.imWidth]
            label = label[:, rs:rs+self.imHeight, cs:cs+self.imWidth]
            labelIndex = labelIndex[:, rs:rs+self.imHeight, cs:cs+self.imWidth ]
            mask = mask[:, rs:rs+self.imHeight, cs:cs+self.imWidth ]

        ## Load data
        # im: input immage batch, Nx3ximHeightximWidth
        # label: binary label of 21 classe, Nx21ximHeightximWidth
        # labelIndex: label of 21 classes, Nx1ximHeightximWidth
        # mask: mask of valid region, Nx1ximHeightximWidth

        batchDict = {
                'im' : im,
                'label': label,
                'labelIndex': labelIndex,
                'mask': mask
                }
        return batchDict


    def loadImage(self, imName ):
        # Load inpute image

        im = Image.open(imName )
        im = np.asarray(im )

        nrows, ncols = im.shape[0], im.shape[1]
        if not (self.imHeight is None or self.imWidth is None ):
            if nrows < self.imHeight or ncols < self.imWidth:
                scaleRow = float(nrows ) / float(self.imHeight )
                scaleCol = float(ncols ) / float(self.imWidth )
                if scaleRow > scaleCol:
                    ncols = int(np.ceil(ncols / scaleCol ) )
                    nrows = int(np.ceil(nrows / scaleCol ) )
                else:
                    ncols = int(np.ceil(ncols / scaleRow ) )
                    nrows = int(np.ceil(nrows / scaleRow ) )
                im = cv2.resize(im, (ncols, nrows), interpolation=cv2.INTER_LINEAR)

        if len(im.shape) == 2:
            print('Warning: load a gray image')
            im = im[:, :, np.newaxis]
            im = np.concatenate([im, im, im], axis=2)
        im = im.astype(np.float32 )  / 255.0

        im = (im - self.imMean ) / self.imStd
        im = im.transpose([2, 0, 1] )
        return im

    def loadLabel(self, labelName ):
        # Load ground-truth label

        labelIndex = Image.open(labelName )
        labelIndex = np.array(labelIndex )
        assert(len(labelIndex.shape ) == 2 )

        nrows, ncols = labelIndex.shape[0], labelIndex.shape[1]
        if not (self.imHeight is None or self.imWidth is None ):
            if nrows < self.imHeight or ncols < self.imWidth:
                scaleRow = float(nrows ) / float(self.imHeight )
                scaleCol = float(ncols ) / float(self.imWidth )
                if scaleRow > scaleCol:
                    ncols = int(np.ceil(ncols / scaleCol ) )
                    nrows = int(np.ceil(nrows / scaleCol ) )
                else:
                    ncols = int(np.ceil(ncols / scaleRow ) )
                    nrows = int(np.ceil(nrows / scaleRow ) )

            labelIndex = cv2.resize(labelIndex, (ncols, nrows), interpolation=cv2.INTER_NEAREST )

        labelIndex = labelIndex.astype(np.int64 )

        nrows, ncols = labelIndex.shape[0], labelIndex.shape[1]
        xIndex, yIndex = np.meshgrid(np.arange(0, ncols), np.arange(0, nrows) )
        xIndex, yIndex, labelIndex = \
                xIndex.astype(np.int32 ), yIndex.astype(np.int32 ), labelIndex.astype(np.int32)

        mask = (labelIndex != 255).astype(np.float32)[np.newaxis, :, :]
        labelIndex[labelIndex == 255] = 0
        labelIndex = labelIndex[np.newaxis, :, :]

        label = np.zeros([self.numClasses, nrows, ncols], dtype = np.float32 )
        label[labelIndex.flatten(), yIndex.flatten(), xIndex.flatten()] = 1.0
        label = label * mask

        return label, labelIndex, mask
