import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.resnet as resnet

class ResBlock(nn.Module ):
    def __init__(self, inch, outch, stride=1, dilation=1 ):
        # Residual Block
        # inch: input feature channel
        # outch: output feature channel
        # stride: the stride of  convolution layer
        super(ResBlock, self ).__init__()
        assert(stride == 1 or stride == 2 )

        self.conv1 = nn.Conv2d(inch, outch, 3, stride, padding = dilation, bias=False,
                dilation = dilation )
        self.bn1 = nn.BatchNorm2d(outch )
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, padding = dilation, bias=False,
                dilation = dilation )
        self.bn2 = nn.BatchNorm2d(outch )

        if inch != outch:
            self.mapping = nn.Sequential(
                        nn.Conv2d(inch, outch, 1, stride, bias=False ),
                        nn.BatchNorm2d(outch )
                    )
        else:
            self.mapping = None

    def forward(self, x ):
        y = x
        if not self.mapping is None:
            y = self.mapping(y )

        out = F.relu(self.bn1(self.conv1(x) ), inplace=True )
        out = self.bn2(self.conv2(out ) )

        out += y
        out = F.relu(out, inplace=True )

        return out

class encoder(nn.Module ):
    def __init__(self ):
        super(encoder, self ).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 2)
        self.b3_2 = ResBlock(256, 256, 1)

        self.b4_1 = ResBlock(256, 512, 2)
        self.b4_2 = ResBlock(512, 512, 1)

    def forward(self, im ):
        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.b4_2(self.b4_1(x4 ) )
        return x1, x2, x3, x4, x5


class decoder(nn.Module ):
    def __init__(self ):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d(512+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(21 )
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(21 )
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.sf = nn.Softmax(dim=1 )

    def forward(self, im, x1, x2, x3, x4, x5):

        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear' )
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x5], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' )
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.sf(self.conv4(y3 ) )

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8) )

        return pred



class encoderDilation(nn.Module ):
    def __init__(self ):
        ## IMPLEMENT YOUR CODE HERE
        super(encoderDilation, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        # 1. "we set the stride of the last 4 residual blocks to be 1 so that 
        #         the highest level of feature maps will be 4x4 times larger 
        #         (compare to basic UNet where stride = 2 and dilation = 1"
        
        # 2. "we set the dilation of residual blocks that are forth and third 
        #     from the end to be 2"
        self.b3_1 = ResBlock(128, 256, stride=1, dilation=2)
        self.b3_2 = ResBlock(256, 256, stride=1, dilation=2)


        # 3. "while the dilation of the residual blocks thar are first and 
        #     second from the end are set to 4"
        self.b4_1 = ResBlock(256, 512, stride=1, dilation=4)
        self.b4_2 = ResBlock(512, 512, stride=1, dilation=4)



    def forward(self, im ):

        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.b4_2(self.b4_1(x4 ) )
        
        return x1, x2, x3, x4, x5


class decoderDilation(nn.Module ):
    def __init__(self, isSpp = False ):
        super(decoderDilation, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(512+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(21 )
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(21 )
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.sf = nn.Softmax(dim=1 )

    def forward(self, im, x1, x2, x3, x4, x5):

        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear' )
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x5], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' )
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.sf(self.conv4(y3 ) )

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8) )


        return pred


# PyramidPooling Block
class PyramidPooling(nn.Module):
    def __init__(self, C, mode='avgpool'):
        super(PyramidPooling, self).__init__()
        self.mode = mode

        # Out PyramidPooling has 4 level with bin sizes = [1, 2, 3, 6]
        # as in the SPPNet paper
        self.conv_1 = nn.Conv2d(in_channels=C, out_channels=int(C/4), kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=C, out_channels=int(C/4), kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=C, out_channels=int(C/4), kernel_size=1, stride=1)
        self.conv_6 = nn.Conv2d(in_channels=C, out_channels=int(C/4), kernel_size=1, stride=1)

    def forward(self, x):
        _, _, H, W = x.size()
        if self.mode == 'avg':
            pool_layer = F.avg_pool2d
        elif self.mode == 'max':
            pool_layer = F.max_pool2d
        
        pooling_1 = pool_layer(x, kernel_size=(H, W))
        pooling_2 = pool_layer(x, kernel_size=(int(H/2)-1, int(W/2)-1) )
        pooling_3 = pool_layer(x, kernel_size=(int(H/3)-1, int(H/3)-1) )
        pooling_6 = pool_layer(x, kernel_size=(int(H/7)+1, int(W/7)+1) )
        
        py_pool_1 = F.interpolate(self.conv_1(pooling_1 ), [H, W], mode='bilinear' )
        py_pool_2 = F.interpolate(self.conv_2(pooling_2 ), [H, W], mode='bilinear' )
        py_pool_3 = F.interpolate(self.conv_3(pooling_3 ), [H, W], mode='bilinear' )
        py_pool_6 = F.interpolate(self.conv_6(pooling_6 ), [H, W], mode='bilinear' )
        
        out = torch.cat([x, py_pool_6, py_pool_3, py_pool_2, py_pool_1], dim=1 )

        return out

class encoderSPP(nn.Module ):
    def __init__(self ):
        super(encoderSPP, self ).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        # 1. "we set the stride of the last 4 residual blocks to be 1 so that 
        #         the highest level of feature maps will be 4x4 times larger 
        #         (compare to basic UNet where stride = 2 and dilation = 1"
        
        # 2. "we set the dilation of residual blocks that are forth and third 
        #     from the end to be 2"
        self.b3_1 = ResBlock(128, 256, stride=1, dilation=2)
        self.b3_2 = ResBlock(256, 256, stride=1, dilation=2)


        # 3. "while the dilation of the residual blocks thar are first and 
        #     second from the end are set to 4"
        self.b4_1 = ResBlock(256, 512, stride=1, dilation=4)
        self.b4_2 = ResBlock(512, 512, stride=1, dilation=4)

        

    def forward(self, im ):

        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.b4_2(self.b4_1(x4 ) )
        
        return x1, x2, x3, x4, x5


class decoderSPP(nn.Module ):
    def __init__(self, isSpp = False ):
        super(decoderSPP, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.pypooling = PyramidPooling(C=512, mode='avg')
        
        
        self.conv1 = nn.Conv2d(1024+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(21 )
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(21 )
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.sf = nn.Softmax(dim=1 )

    def forward(self, im, x1, x2, x3, x4, x5):

        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        x6 = self.pypooling(x5)
        
        x6 = F.interpolate(x6, [nh, nw], mode='bilinear' )
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x6], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' )
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.sf(self.conv4(y3 ) )

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8) )


        return pred



def loadPretrainedWeight(network, isOutput = False ):
    paramList = []
    resnet18 = resnet.resnet18(pretrained=True )
    for param in resnet18.parameters():
        paramList.append(param )

    cnt = 0
    for param in network.parameters():
        if paramList[cnt ].size() == param.size():
            param.data.copy_(paramList[cnt].data )
            if isOutput:
                print(param.size() )
        else:
            print(param.shape, paramList[cnt].shape )
            break
        cnt += 1
