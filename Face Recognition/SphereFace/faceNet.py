import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
np.warnings.filterwarnings('ignore')


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4 ):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / torch.clamp(xlen.view(-1,1) * wlen.view(1,-1), min=1e-8)
        cos_theta = cos_theta.clamp(-1,1)

        # IMPLEMENT phi_theta
        # Use multi-angle formulae to calculte cos(m*theta)
        # these fornulae given in self.mlambda
        cos_m_theta = self.mlambda[self.m](cos_theta)
        # Calculate the hyper-parameter k
        # k should be in the [0, m-1] and
        # theta should be in [k*pi/m, (k+1)*pi/m]
        # here in the forward process, we choose k that makes theta = k*pi/m
        theta = Variable(cos_theta.data.acos())
        k = (self.m * theta / np.pi).floor()
        # Finally, calculate phi using the equation in SphereFace paper
        phi_theta = (-1) ** (k) * cos_m_theta - 2 * k


        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)

        output = (cos_theta,phi_theta)
        return output


class CustomLoss(nn.Module):
    def __init__(self ):
        super(CustomLoss, self).__init__()

        # Parameters for computing loss function
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        # IMPLEMENT loss
        # Construct a tensor with index-i equals 1 ohterwise 0
        #    Initialize with all 0 in index tensor, then fill in 1 in corresponding places
        index = cos_theta * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        #    Convert index to int for further use
        index = Variable(index.to(torch.uint8))


        # Notice the difference between the angular softmax and regular softmax
        # Prepare the input for softmax by replace cos_function with phi-function
        #   at index-i (already calculated in index)
        output = cos_theta * 1.0
        #   Here, it turns out to be crucial to use the annealing optimization
        #    in Appendix G in SphereFace Paper (for more discussion, see the assignment report)
        #   We use lambda starting from LambdaMax and gradually decay to LambaMin
        #   for the decay rate we choose the one based on https://github.com/clcarwin/sphereface_pytorch
        lamb = max(self.LambdaMin, self.LambdaMax / (1+0.1*self.it))
        # Implement annealing
        output[index] -= cos_theta[index] / (1. + lamb)
        output[index] += phi_theta[index] / (1. + lamb)


        # Use softmax from Pytorch for loss function
        logpt = F.log_softmax(output)
        # Calculate the loss functiong use eq.(7) in SphereFace paper
        loss = (-logpt.gather(1, target).view(-1)).mean()



        _, predictedLabel = torch.max(cos_theta.data, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(target.data).cpu().sum().item() ) / float(target.size(0) )

        return loss, accuracy



class faceNet(nn.Module):
    def __init__(self,classnum=10574, feature=False, m = 1.35):
        super(faceNet, self).__init__()
        self.classnum = classnum
        self.feature = feature

        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.bn2_4 = nn.BatchNorm2d(128)
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.bn2_5 = nn.BatchNorm2d(128)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.bn3_4 = nn.BatchNorm2d(256)
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.bn3_5 = nn.BatchNorm2d(256)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.bn3_6 = nn.BatchNorm2d(256)
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.bn3_7 = nn.BatchNorm2d(256)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.bn3_8 = nn.BatchNorm2d(256)
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.bn3_9 = nn.BatchNorm2d(256)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = CustomLinear(in_features = 512,
                out_features = self.classnum, m=m)


    def forward(self, x):
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = x + self.relu1_3(self.bn1_3(self.conv1_3(self.relu1_2(self.bn1_2(self.conv1_2(x))))))

        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = x + self.relu2_3(self.bn2_3(self.conv2_3(self.relu2_2(self.bn2_2(self.conv2_2(x))))))
        x = x + self.relu2_5(self.bn2_5(self.conv2_5(self.relu2_4(self.bn2_4(self.conv2_4(x))))))

        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = x + self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(x))))))
        x = x + self.relu3_5(self.bn3_5(self.conv3_5(self.relu3_4(self.bn3_4(self.conv3_4(x))))))
        x = x + self.relu3_7(self.bn3_7(self.conv3_7(self.relu3_6(self.bn3_6(self.conv3_6(x))))))
        x = x + self.relu3_9(self.bn3_9(self.conv3_9(self.relu3_8(self.bn3_8(self.conv3_8(x))))))

        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = x + self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(x)) ) )) )

        x = x.view(x.size(0),-1)
        x = self.bn5(self.fc5(x))

        if self.feature:
            return x

        x = self.fc6(x)
        return x
