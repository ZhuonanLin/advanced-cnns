import torch
from torch.autograd import Variable
import torch.functional as F
from dataLoader import BatchLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io
import model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='basic', type=str)
parser.add_argument('--imageRoot', default='/datasets/cs252csp19-public/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cs252csp19-public/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cs252csp19-public/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--experiment', default='checkpoint', help='the path to store sampled images and models' )
parser.add_argument('--epoches', type=int, default=24, help='the number of epochs being trained')
parser.add_argument('--batchSize', type=int, default=32, help='the size of a batch')
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes')
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--num_workers', default=0, help='number of workers used in PyTorch')
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')
parser.add_argument('--imHeight', default=300, type=int)
parser.add_argument('--imWidth', default=300, type=int)
parser.add_argument('--iterationDecreaseLR', default=800, type=int)
parser.add_argument('--iterationEnd', default=3000)


# The detail network setting
opt = parser.parse_args()
print(opt)

# Not necessary for the training process
colormap = io.loadmat(opt.colormap )['cmap']



if opt.isSpp == True :
    opt.isDilation = False

if opt.isDilation:
    opt.experiment += '_dilation'
if opt.isSpp:
    opt.experiment += '_spp'

os.mkdir(opt.experiment)

# Initialize network
if opt.isDilation:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation()
elif opt.isSpp:
    encoder = model.encoderSPP()
    decoder = model.decoderSPP()
else:
    encoder = model.encoder()
    decoder = model.decoder()




model.loadPretrainedWeight(encoder, False)

encoder = encoder.cuda(opt.gpuId)
decoder = decoder.cuda(opt.gpuId)


# Initialize variables for training process
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imHeight, opt.imWidth) )
labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, opt.imHeight, opt.imWidth) )
maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imHeight, opt.imWidth) )
labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, opt.imHeight, opt.imWidth) )

# Put variables on GPU
if not opt.noCuda:
    imBatch = imBatch.cuda(opt.gpuId)
    labelBatch = labelBatch.cuda(opt.gpuId)
    labelIndexBatch = labelIndexBatch.cuda(opt.gpuId)
    maskBatch = maskBatch.cuda(opt.gpuId)
    encoder = encoder.cuda(opt.gpuId)
    decoder = decoder.cuda(opt.gpuId)


# Load data
segDataset = BatchLoader(imageRoot = opt.imageRoot, 
                    labelRoot = opt.labelRoot, 
                    fileList = opt.fileList,
                    imWidth = opt.imWidth, imHeight = opt.imHeight)
segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=opt.num_workers, shuffle=True)

# function to calculate mIoU accuracy during training
def calculate_mIoU_accuracy(pred, labelBatch, labelIndexBatch, maskBatch, numClasses):
    '''
    Calculage mIoU during training, given the temporary results during during

    code mainly borrowd from test.py
    '''
    confcounts = np.zeros( (numClasses, numClasses), dtype=np.int64 )
    accuracy = np.zeros(numClasses, dtype=np.float32 )
    # Compute mean IOU
    hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch )
    confcounts += hist

    for n in range(0, numClasses):
        rowSum = np.sum(confcounts[n, :] )
        colSum = np.sum(confcounts[:, n] )
        interSum = confcounts[n, n]
        accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5)

    meanAccuracy = np.mean(accuracy )
    
    return meanAccuracy


# Set up optimizer, we use deffirent learning rate for encoder (small) and decoder (large)
lr_encoder = 1e-4
lr_decoder = 1e-2

optimizer = optim.Adam([
                {'params': encoder.parameters(), 'lr': lr_encoder},
                {'params': decoder.parameters(), 'lr': lr_decoder}
                ]) # encoder has lr=1e-3 and decoder has 1e-1


iterationDecreaseLR = opt.iterationDecreaseLR


lossArr = []
accuracyArr = []
iteration = 0

#lossFunction = torch.mean()


for epoch in range(opt.epoches):
    print('Starting epoch {}/{}.'.format(epoch+1, opt.epoches))
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')

    for i, dataBatch in enumerate(segLoader):
        iteration += 1

        # Read data
        image_cpu = dataBatch['im']
        imBatch.data.resize_(image_cpu.size() )
        imBatch.data.copy_(image_cpu )

        label_cpu = dataBatch['label']
        labelBatch.data.resize_(label_cpu.size() )
        labelBatch.data.copy_(label_cpu )
        
        labelIndex_cpu = dataBatch['labelIndex' ]
        labelIndexBatch.data.resize_(labelIndex_cpu.size() )
        labelIndexBatch.data.copy_(labelIndex_cpu )

        mask_cpu = dataBatch['mask' ]
        maskBatch.data.resize_( mask_cpu.size() )
        maskBatch.data.copy_( mask_cpu )


        # Train network
        optimizer.zero_grad()

        # Feed forward
        x1, x2, x3, x4, x5 = encoder(imBatch)
        pred = decoder(imBatch, x1, x2, x3, x4, x5)
        
        # Calculate Loss and back propergation
        #loss = lossFunction( pred * labelBatch )
        loss = torch.mean(pred * labelBatch)
        loss.backward()
        
        # Calculate mIoU accuracy each iteration as a mark to monitor the training process
        accuracy = calculate_mIoU_accuracy(pred, labelBatch, labelIndexBatch, maskBatch, opt.numClasses)
        
        # update training step
        optimizer.step()
        
        # Record training progress
        lossArr.append(loss.cpu().data.item())
        accuracyArr.append(accuracy)


        # Calculate mean training loss (pixel-wise cross-entropy) and mean training accuracy (mIoU)
        if iteration >= 200:
            meanLoss = np.mean(np.array(lossArr[-1000:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[-1000:] ) )
        else:
            meanLoss = np.mean(np.array(lossArr[:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[:] ) )
        
        print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f' % (epoch+1, iteration, lossArr[-1], meanLoss ) )
        print('Epoch %d iteration %d: Accura %.5f Accumulated Accura %.5f' % (epoch+1, iteration, accuracyArr[-1], meanAccuracy ) )
        trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' % (epoch+1, iteration, lossArr[-1], meanLoss ) )
        trainingLog.write('Epoch %d iteration %d: Accura %.5f Accumulated Accura %.5f\n' % (epoch+1, iteration, accuracyArr[-1], meanAccuracy ) )
        if iteration == iterationDecreaseLR and lr_encoder > 1e-4:
            print('The learning rate is being decreased at iteration %d' % iteration )
            trainingLog.write('The learning rate is being decreased at iteration %d\n' % iteration )
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
                print('New learning rate = %f' % (param_group['lr']))
            iterationDecreaseLR *= 2
            lr_encoder /= 5.
            lr_decoder /= 5.
    
    
    print('-' * 80)



    if iteration == opt.iterationEnd:
        np.save('%s/loss.npy' % opt.experiment, np.array(lossArr ) )
        np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracyArr ) )
        torch.save(encoder.state_dict(), '%s/encoderFinal_%d.pth' % (opt.experiment, epoch+1) )
        torch.save(decoder.state_dict(), '%s/decoderFinal_%d.pth' % (opt.experiment, epoch+1) )
        break

    trainingLog.close()

    if iteration >= opt.iterationEnd:
        break

    if (epoch+1) % 2 == 0:
        np.save('%s/loss.npy' % opt.experiment, np.array(lossArr ) )
        np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracyArr ) )
        torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch+1) )
        torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch+1) )
