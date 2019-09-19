import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import faceNet
import torch.nn as nn
import os
import numpy as np


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--imageRoot', default='../CASIA-WebFace/', help='path to input images')
parser.add_argument('--alignmentRoot', default='./data/casia_landmark.txt', help='path to the alignment file')
parser.add_argument('--experiment', default='checkpoint', help='the path to store sampled images and models')
parser.add_argument('--marginFactor', type=float, default=0.35, help='margin factor')
parser.add_argument('--scaleFactor', type=float, default=30, help='scale factor')
parser.add_argument('--imHeight', type=int, default=112, help='height of input image')
parser.add_argument('--imWidth', type=int, default=96, help='width of input image')
parser.add_argument('--batchSize', type=int, default=128, help='the size of a batch')
parser.add_argument('--nepoch', type=int, default=20, help='the training epoch')
parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
parser.add_argument('--iterationDecreaseLR', type=int, nargs='+', default=[16000, 24000], help='the iteration to decrease learning rate')
parser.add_argument('--iterationEnd', type=int, default=28000, help='the iteration to end training')

# The detail network setting
opt = parser.parse_args()
print(opt)

# Save all the codes
os.system('mkdir %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initialize image batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imHeight, opt.imWidth) )
targetBatch = Variable(torch.LongTensor(opt.batchSize, 1) )

# Initialize network
net = faceNet.faceNet(m = opt.marginFactor, feature = False )
lossLayer = faceNet.CustomLoss(s = opt.scaleFactor )

# Move network and containers to gpu
if not opt.noCuda:
    imBatch = imBatch.cuda(opt.gpuId )
    targetBatch = targetBatch.cuda(opt.gpuId )
    net = net.cuda(opt.gpuId )

# Initialize optimizer
optimizer = optim.SGD(net.parameters(), lr=opt.initLR, momentum=0.9, weight_decay=5e-4 )

# Initialize dataLoader
faceDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        alignmentRoot = opt.alignmentRoot,
        cropSize = (opt.imWidth, opt.imHeight )
        )
faceLoader = DataLoader(faceDataset, batch_size = opt.batchSize, num_workers = 16, shuffle = False )

lossArr = []
accuracyArr = []
iteration = 0
for epoch in range(0, opt.nepoch ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, dataBatch in enumerate(faceLoader ):
        iteration += 1

        # Read data
        image_cpu = dataBatch['img']
        imBatch.data.resize_(image_cpu.size() )
        imBatch.data.copy_(image_cpu )

        target_cpu = dataBatch['target']
        targetBatch.data.resize_(target_cpu.size() )
        targetBatch.data.copy_(target_cpu )

        # Train network
        optimizer.zero_grad()

        pred = net(imBatch )
        loss, accuracy = lossLayer(pred, targetBatch )
        loss.backward()

        optimizer.step()

        # Output the log information
        lossArr.append(loss.cpu().data.item() )
        accuracyArr.append(accuracy )

        if iteration >= 1000:
            meanLoss = np.mean(np.array(lossArr[-1000:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[-1000:] ) )
        else:
            meanLoss = np.mean(np.array(lossArr[:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[:] ) )

        print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f' % (epoch, iteration, lossArr[-1], meanLoss ) )
        print('Epoch %d iteration %d: Accura %.5f Accumulated Accura %.5f' % (epoch, iteration, accuracyArr[-1], meanAccuracy ) )
        trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' % (epoch, iteration, lossArr[-1], meanLoss ) )
        trainingLog.write('Epoch %d iteration %d: Accura %.5f Accumulated Accura %.5f\n' % (epoch, iteration, accuracyArr[-1], meanAccuracy ) )
        if iteration == 1:
            vutils.save_image( 0.5 * (imBatch.data+1), '%s/images.png' % (opt.experiment ) )

        if iteration in opt.iterationDecreaseLR:
            print('The learning rate is being decreased at iteration %d' % iteration )
            trainingLog.write('The learning rate is being decreased at iteration %d\n' % iteration )
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        if iteration == opt.iterationEnd:
            np.save('%s/loss.npy' % opt.experiment, np.array(lossArr ) )
            np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracyArr ) )
            torch.save(net.state_dict(), '%s/netFinal_%d.pth' % (opt.experiment, epoch+1) )
            break

    trainingLog.close()

    if iteration >= opt.iterationEnd:
        break

    if (epoch+1) % 2 == 0:
        np.save('%s/loss.npy' % opt.experiment, np.array(lossArr ) )
        np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracyArr ) )
        torch.save(net.state_dict(), '%s/net_%d.pth' % (opt.experiment, epoch+1) )
