import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_modules import Up3d, DoubleConv3d, Down3d, OutConv3d, \
                            Up ,DoubleConv2dShift, Down2dShift, OutConv2dShift, \
                            DoubleConv, Down, OutConv

def initial_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Conv1d) or isinstance(m,nn.Conv3d):
        nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight,0,'fan_in','relu')
        nn.init.constant_(m.bias,0) if m.bias is not None else None
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0) if m.bias is not None else None
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0) if m.bias is not None else None

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()

    def freeze(self):
        for k,p in self.named_parameters():
            p.requires_grad = False

    def get_parameters(self):
        param = []
        for k,p in self.named_parameters():
            if p.requires_grad:
                param.append(p)
        return param

    def initial(self, pretrained_weights=None):
        if pretrained_weights is None:
            self.apply(initial_weights)
        else:
            ckpt = torch.load(pretrained_weights, map_location='cpu')
            self.load_state_dict(ckpt, strict=False)
        return self


from models.resnet_modules import BasicBlock
import math
class UNetResNet18(BasicModule):
    def __init__(self, n_classes, n_segment=2):
        super(UNetResNet18, self).__init__()
        self.n_classes = n_classes
        self.n_segment = n_segment

        block = BasicBlock
        conv1_channels = 32
        layers = [2,2,2,2]
        channels = [conv1_channels*n_segment,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)

        self.up1 = Up(channels[3]+channels[2], up_channels[0])
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1])
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2])
        self.up4 = Up(conv1_channels*n_segment+up_channels[2],up_channels[3])

        self.outc = OutConv(up_channels[3], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        # print('x1 = ', x1.shape)
        # x1 [b, 64, 224, 224]
        # [bt, c, h, w] -> [b, tc, h , w]
        t = self.n_segment
        x1 = x1.reshape(tuple([-1, t*x1.shape[1]]) + x1.shape[2:])
        x2 = self.layer1(x1)
        # print('x2 = ', x2.shape)
        # x2 [b, 64, 112, 112]
        x3 = self.layer2(x2)
        # print('x3 = ', x3.shape)
        # x3 [b, 128, 56, 56]
        x4 = self.layer3(x3)
        # x4 [b, 256, 28, 28]
        # print('x4 = ', x4.shape)
        x5 = self.layer4(x4)
        # print('x5 = ', x5.shape)
        # x5 [b, 512, 14, 14]
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,28,28]
        # print('x = ', x.shape)
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,56,56]
        # print('x = ', x.shape)
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,112,112]
        # print('x = ', x.shape)
        x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,224,224]
        # print('x = ', x.shape)
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[b,c,t,h,w]"""
        # [b,c,t,h,w]->[b,t,c,h,w]
        x0 = x.permute(0, 2, 1, 3, 4)
        # [b,t,c,h,w]->[bt,c,h,w]
        x = x0.reshape(tuple([-1]) + x0.shape[-3:])
        return self._forward_mlu_(x)

    @torch.no_grad()
    def inference(self, x):
        return self.sigmoid(self.forward(x))


class UNetResNet18_MLU(nn.Module):
    def __init__(self, n_classes, n_segment=2):
        super(UNetResNet18_MLU, self).__init__()
        self.n_classes = n_classes
        self.n_segment = n_segment

        block = BasicBlock
        conv1_channels = 32
        layers = [2,2,2,2]
        channels = [conv1_channels*n_segment,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)

        self.up1 = Up(channels[3]+channels[2], up_channels[0])
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1])
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2])
        self.up4 = Up(conv1_channels*n_segment+up_channels[2],up_channels[3])

        self.outc = OutConv(up_channels[3], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        x1 = self.conv1(x)
        # [bt, c, h, w] -> [b, tc, h , w]
        t = self.n_segment
        x1 = x1.reshape(tuple([-1, t*x1.shape[1]]) + x1.shape[2:])
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,28,28]
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,56,56]
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,112,112]
        x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,224,224]
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[bt,c,h,w]"""
        return self.sigmoid(self._forward_mlu_(x))


class UNetResNet18_BN(BasicModule):
    def __init__(self, n_classes, n_segment=2, use_bn=False):
        super(UNetResNet18_BN, self).__init__()
        self.n_classes = n_classes
        self.n_segment = n_segment

        block = BasicBlock
        conv1_channels = 32
        layers = [2,2,2,2]
        channels = [conv1_channels*n_segment,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64

        print('use_bn =', use_bn)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, use_bn=use_bn)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, use_bn=use_bn)

        self.up1 = Up(channels[3]+channels[2], up_channels[0], use_bn=use_bn)
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1], use_bn=use_bn)
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2], use_bn=use_bn)
        self.up4 = Up(conv1_channels*n_segment+up_channels[2],up_channels[3], use_bn=use_bn)

        self.outc = OutConv(up_channels[3], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_bn=True):
        # print('(_make_layer)use_bn =', use_bn)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn = use_bn))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        # print('x1 = ', x1.shape)
        # x1 [b, 64, 224, 224]
        # [bt, c, h, w] -> [b, tc, h , w]
        t = self.n_segment
        x1 = x1.reshape(tuple([-1, t*x1.shape[1]]) + x1.shape[2:])
        x2 = self.layer1(x1)
        # print('x2 = ', x2.shape)
        # x2 [b, 64, 112, 112]
        x3 = self.layer2(x2)
        # print('x3 = ', x3.shape)
        # x3 [b, 128, 56, 56]
        x4 = self.layer3(x3)
        # x4 [b, 256, 28, 28]
        # print('x4 = ', x4.shape)
        x5 = self.layer4(x4)
        # print('x5 = ', x5.shape)
        # x5 [b, 512, 14, 14]
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,28,28]
        # print('x = ', x.shape)
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,56,56]
        # print('x = ', x.shape)
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,112,112]
        # print('x = ', x.shape)
        x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,224,224]
        # print('x = ', x.shape)
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[b,c,t,h,w]"""
        # [b,c,t,h,w]->[b,t,c,h,w]
        x0 = x.permute(0, 2, 1, 3, 4)
        # [b,t,c,h,w]->[bt,c,h,w]
        x = x0.reshape(tuple([-1]) + x0.shape[-3:])
        return self._forward_mlu_(x)

    @torch.no_grad()
    def inference(self, x):
        return self.sigmoid(self.forward(x))


class UNetResNet18_BN_MLU(nn.Module):
    def __init__(self, n_classes, n_segment=2, use_bn=False):
        super(UNetResNet18_BN_MLU, self).__init__()
        self.n_classes = n_classes
        self.n_segment = n_segment

        block = BasicBlock
        conv1_channels = 32
        layers = [2,2,2,2]
        channels = [conv1_channels*n_segment,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, use_bn=use_bn)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, use_bn=use_bn)

        self.up1 = Up(channels[3]+channels[2], up_channels[0], use_bn=use_bn)
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1], use_bn=use_bn)
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2], use_bn=use_bn)
        self.up4 = Up(conv1_channels*n_segment+up_channels[2],up_channels[3], use_bn=use_bn)

        self.outc = OutConv(up_channels[3], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_bn=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn=use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn = use_bn))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        x1 = self.conv1(x)
        # [bt, c, h, w] -> [b, tc, h , w]
        t = self.n_segment
        x1 = x1.reshape(tuple([-1, t*x1.shape[1]]) + x1.shape[2:])
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,28,28]
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,56,56]
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,112,112]
        x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,224,224]
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[bt,c,h,w]"""
        return self.sigmoid(self._forward_mlu_(x))



from models.unet_modules import BasicBlock as BasicBlock_TSM
class UNetResNet18_Shift(BasicModule):
    def __init__(self, n_classes, n_segment=None):
        super(UNetResNet18_Shift, self).__init__()
        self.n_classes = n_classes
        self.n_segment = n_segment

        block = BasicBlock_TSM
        layers = [2,2,2,2]
        channels = [64,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64

        conv1_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, n_segment=n_segment)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, n_segment=n_segment)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, n_segment=n_segment)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, n_segment=n_segment)

        self.up1 = Up(channels[3]+channels[2], up_channels[0])
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1])
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2])
        self.up4 = Up(conv1_channels+up_channels[2],up_channels[3])

        self.outc = OutConv(up_channels[3], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, n_segment=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_segment))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_segment=n_segment))

        return nn.Sequential(*layers)

    def temperal_pooling(self, x ):
        """
        :param x: [bt,c,h,w] 
        :return: [b,c,h,w] 
        """
        #[bt,c,h,w]->[b,t,c,h,w]
        x = x.reshape(tuple([-1, self.n_segment])+x.shape[1:])
        #[b,t,c,h,w]->[b,1,c,h,w]->[b,c,h,w]
        return torch.mean(x,dim=1).squeeze(dim=1)

    def _forward_mlu_(self,x):
        """[b,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        # print('x1 = ', x1.shape)
        # x1 [b, 64, 224, 224]
        x2 = self.layer1(x1)
        # print('x2 = ', x2.shape)
        # x2 [b, 64, 112, 112]
        x3 = self.layer2(x2)
        # print('x3 = ', x3.shape)
        # x3 [b, 128, 56, 56]
        x4 = self.layer3(x3)
        # x4 [b, 256, 28, 28]
        # print('x4 = ', x4.shape)
        x5 = self.layer4(x4)
        # print('x5 = ', x5.shape)
        # x5 [b, 512, 14, 14]
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,28,28]
        # print('x = ', x.shape)
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,56,56]
        # print('x = ', x.shape)
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,112,112]
        # print('x = ', x.shape)
        x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,224,224]
        # print('x = ', x.shape)
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[b,c,t,h,w]"""
        #[b,c,t,h,w]->[b,t,c,h,w]
        x0 = x.permute(0,2,1,3,4)
        #[b,t,c,h,w]->[bt,c,h,w]
        x = x0.reshape( tuple([-1]) + x0.shape[-3:])
        x = self._forward_mlu_(x)
        return self.temperal_pooling(x) #[b,c,h,w]

    @torch.no_grad()
    def inference(self, x):
        return self.sigmoid(self.forward(x))


class UNetResNet18_Shift_MLU(BasicModule):
    def __init__(self, n_classes, n_segment=None):
        super(UNetResNet18_Shift_MLU, self).__init__()
        self.n_classes = n_classes
        self.n_segment = n_segment

        block = BasicBlock_TSM
        layers = [2,2,2,2]
        channels = [64,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64

        conv1_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, n_segment=n_segment)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, n_segment=n_segment)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, n_segment=n_segment)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, n_segment=n_segment)

        self.up1 = Up(channels[3]+channels[2], up_channels[0])
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1])
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2])
        self.up4 = Up(conv1_channels+up_channels[2],up_channels[3])

        self.outc = OutConv(up_channels[3], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, n_segment=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_segment))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_segment=n_segment))

        return nn.Sequential(*layers)

    def temperal_pooling(self, x ):
        """
        :param x: [bt,c,h,w] 
        :return: [b,c,h,w] 
        """
        #[bt,c,h,w]->[b,t,c,h,w]
        x = x.reshape(tuple([-1, self.n_segment])+x.shape[1:])
        #[b,t,c,h,w]->[b,1,c,h,w]->[b,c,h,w]
        return torch.mean(x,dim=1)

    def _forward_mlu_(self,x):
        """[b,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,28,28]
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,56,56]
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,112,112]
        x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,224,224]
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[bt,c,h,w]"""
        return self.sigmoid(self.temperal_pooling(self._forward_mlu_(x)))



"""
UNetResNet18_BN_Scaled 224 --> 56
输入输出不保持一致
"""
class UNetResNet18_BN_Scaled(BasicModule):
    def __init__(self, n_classes, use_bn=False):
        super(UNetResNet18_BN_Scaled, self).__init__()
        self.n_classes = n_classes

        block = BasicBlock
        conv1_channels = 64
        layers = [2,2,2,2]
        channels = [conv1_channels,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64

        print('use_bn =', use_bn)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, use_bn=use_bn)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, use_bn=use_bn)

        # self.downs = []
        # for ch,num_layer in zip(channels, layers):
        #     self.downs.append( self._make_layer(block, ch, num_layer , stride=2, use_bn=use_bn) )

        # self.ups = []
        # self.ups.append( Up(channels[3]+channels[2], up_channels[0], use_bn=use_bn) )
        # self.ups.append( Up(channels[1]+up_channels[0], up_channels[1], use_bn=use_bn) )
        # self.ups.append( Up(channels[0]+up_channels[1], up_channels[2], use_bn=use_bn) )
        # self.ups.append( Up(conv1_channels+up_channels[2],up_channels[3], use_bn=use_bn) )
        self.up1 = Up(channels[3]+channels[2], up_channels[0], use_bn=use_bn)
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1], use_bn=use_bn)
        self.up3 = Up(channels[0]+up_channels[1], up_channels[2], use_bn=use_bn)
        # self.up4 = Up(conv1_channels+up_channels[2],up_channels[3], use_bn=use_bn)

        self.outc = OutConv(up_channels[2], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_bn=True):
        # print('(_make_layer)use_bn =', use_bn)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn = use_bn))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        # print('x1 = ', x1.shape)
        # x1 [b, 64, 112, 112]
        x2 = self.layer1(x1)
        # print('x2 = ', x2.shape)
        # x2 [b, 64, 56, 56]
        x3 = self.layer2(x2)
        # print('x3 = ', x3.shape)
        # x3 [b, 128, 28, 28]
        x4 = self.layer3(x3)
        # x4 [b, 256, 14, 14]
        # print('x4 = ', x4.shape)
        x5 = self.layer4(x4)
        # print('x5 = ', x5.shape)
        # x5 [b, 512, 7, 7]
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,14,14]
        # print('x = ', x.shape)
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,28,28]
        # print('x = ', x.shape)
        x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,56,56]
        # print('x = ', x.shape)
        # x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,112,112]
        # print('x = ', x.shape)
        logits = self.outc(x) #[b,c,h,w]z
        return logits

    def forward(self, x):
        # [b,c,t,h,w]->[b,t,c,h,w]
        # x0 = x.permute(0, 2, 1, 3, 4)
        # [b,t,c,h,w]->[bt,c,h,w]
        # x = x0.reshape(tuple([-1]) + x0.shape[-3:])
        return self._forward_mlu_(x)

    @torch.no_grad()
    def inference(self, x):
        return self.sigmoid(self.forward(x))

"""
UNetResNet18_BN_Scaled28 224->28
"""
class UNetResNet18_BN_Scaled28(BasicModule):
    def __init__(self, n_classes, use_bn=False):
        super(UNetResNet18_BN_Scaled28, self).__init__()
        self.n_classes = n_classes

        block = BasicBlock
        conv1_channels = 64
        layers = [2,2,2,2]
        channels = [conv1_channels,128,256,512]
        up_channels = [256,128,64,32]
        self.inplanes = 64

        print('use_bn =', use_bn)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, use_bn=use_bn)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, use_bn=use_bn)

        self.up1 = Up(channels[3]+channels[2], up_channels[0], use_bn=use_bn)
        self.up2 = Up(channels[1]+up_channels[0], up_channels[1], use_bn=use_bn)
        # self.up3 = Up(channels[0]+up_channels[1], up_channels[2], use_bn=use_bn)
        # self.up4 = Up(conv1_channels+up_channels[2],up_channels[3], use_bn=use_bn)

        self.outc = OutConv(up_channels[1], n_classes)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_bn=True):
        # print('(_make_layer)use_bn =', use_bn)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn = use_bn))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        # print('x1 = ', x1.shape)
        # x1 [b, 64, 112, 112]
        x2 = self.layer1(x1)
        # print('x2 = ', x2.shape)
        # x2 [b, 64, 56, 56]
        x3 = self.layer2(x2)
        # print('x3 = ', x3.shape)
        # x3 [b, 128, 28, 28]
        x4 = self.layer3(x3)
        # x4 [b, 256, 14, 14]
        # print('x4 = ', x4.shape)
        x5 = self.layer4(x4)
        # print('x5 = ', x5.shape)
        # x5 [b, 512, 7, 7]
        x = self.up1(x5, x4) #[,512+256,...] ->[b,256,14,14]
        # print('x = ', x.shape)
        x = self.up2(x, x3)  #[,256+128,...] ->[b,128,28,28]
        # print('x = ', x.shape)
        # x = self.up3(x, x2)  #[,128+64,...]  ->[b,64,56,56]
        # print('x = ', x.shape)
        # x = self.up4(x, x1)  #[,64+64,...]  ->[b,32,112,112]
        # print('x = ', x.shape)
        logits = self.outc(x) #[b,c,h,w]z
        return logits

    def forward(self, x):
        # [b,c,t,h,w]->[b,t,c,h,w]
        # x0 = x.permute(0, 2, 1, 3, 4)
        # [b,t,c,h,w]->[bt,c,h,w]
        # x = x0.reshape(tuple([-1]) + x0.shape[-3:])
        return self._forward_mlu_(x)

    @torch.no_grad()
    def inference(self, x):
        return self.sigmoid(self.forward(x))





class ResNet34_BN_Scaled(BasicModule):
    def __init__(self, n_classes, use_bn=False):
        super(ResNet34_BN_Scaled, self).__init__()
        self.n_classes = n_classes

        block = BasicBlock
        conv1_channels = 64
        layers = [3,4,6,3]
        channels = [conv1_channels,128,256,512]
        strides = [2,2,2,1]
        self.inplanes = 64

        print('use_bn =', use_bn)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], use_bn=use_bn)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], use_bn=use_bn)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], use_bn=use_bn)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], use_bn=use_bn)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, conv1_channels, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(conv1_channels),
        #     nn.LeakyReLU(inplace=True),
        # )
        #
        # self.layers = []
        # for nch, nlayer, nstride in zip(channels, layers, strides):
        #     self.layers.append( self._make_layer(block, nch, nlayer, stride=nstride, use_bn=use_bn) )

        self.outc = OutConv(channels[-1], n_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_bn=True):
        # print('(_make_layer)use_bn =', use_bn)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn = use_bn))

        return nn.Sequential(*layers)


    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        # [b,3,224,224]
        x1 = self.conv1(x)
        # print('x1 = ', x1.shape)
        # x1 [b, 64, 112, 112]
        x2 = self.layer1(x1)
        # print('x2 = ', x2.shape)
        # x2 [b, 64, 56, 56]
        x3 = self.layer2(x2)
        # print('x3 = ', x3.shape)
        # x3 [b, 128, 28, 28]
        x4 = self.layer3(x3)
        # x4 [b, 256, 14, 14]
        # print('x4 = ', x4.shape)
        x5 = self.layer4(x4)
        x5 = self.dropout(x5)

        logits = self.outc(x5) #[b,c,h,w]z
        return logits

    def forward(self, x):
        # [b,c,t,h,w]->[b,t,c,h,w]
        # x0 = x.permute(0, 2, 1, 3, 4)
        # [b,t,c,h,w]->[bt,c,h,w]
        # x = x0.reshape(tuple([-1]) + x0.shape[-3:])
        return self._forward_mlu_(x)

    @torch.no_grad()
    def inference(self, x):
        return self.sigmoid(self.forward(x))





class UNet2DShiftMLU(nn.Module):
    def __init__(self, n_channels, n_classes, n_segment):
        super(UNet2DShiftMLU, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.bilinear = bilinear
        self.n_segment = n_segment#With format [B,C,T,H,W], n_segment=T

        self.inc = DoubleConv2dShift(n_channels, 32, n_segment)
        self.down1 = Down2dShift(32, 64, n_segment)
        self.down2 = Down2dShift(64, 128, n_segment)
        self.down3 = Down2dShift(128, 256, n_segment)
        self.down4 = Down2dShift(256, 256, n_segment)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv2dShift(32, n_classes, n_segment)

        self.last_layer = nn.Sigmoid()

    def temperal_pooling(self, x ):
        """
        :param x: [bt,c,h,w] 
        :return: [bt,c,h,w] 
        """
        #[bt,c,h,w]->[b,t,c,h,w]
        x = x.reshape(tuple([-1, self.n_segment])+x.shape[1:])
        # print(x.shape)
        #[b,t,c,h,w]->[b,c,h,w]
        return torch.mean(x,dim=1)

    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) #[bt,c,h,w]
        # print('logits.shape=', logits.shape)
        logits = self.temperal_pooling(logits)#[b,c,h,w]
        print('<-- _forward_mlu_')
        return logits

    def forward(self, x):
        """[bt,c,h,w]"""
        x = self._forward_mlu_(x)
        # print(x.shape)
        return self.last_layer(x)




class UNet2DShift(BasicModule):
    def __init__(self, n_channels, n_classes, n_segment):
        super(UNet2DShift, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.bilinear = bilinear
        self.n_segment = n_segment#With format [B,C,T,H,W], n_segment=T

        self.inc = DoubleConv2dShift(n_channels, 32, n_segment)
        self.down1 = Down2dShift(32, 64, n_segment)
        self.down2 = Down2dShift(64, 128, n_segment)
        self.down3 = Down2dShift(128, 256, n_segment)
        self.down4 = Down2dShift(256, 256, n_segment)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv2dShift(32, n_classes, n_segment)

    def temperal_pooling(self, x ):
        """
        :param x: [bt,c,h,w] 
        :return: [bt,c,h,w] 
        """
        #[bt,c,h,w]->[b,t,c,h,w]
        x = x.reshape(tuple([-1, self.n_segment])+x.shape[1:])
        #[b,t,c,h,w]->[b,1,c,h,w]->[b,c,h,w]
        return torch.mean(x,dim=1).squeeze(dim=1)

    def _forward_mlu_(self,x):
        """[bt,c,h,w]"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) #[bt,c,h,w]
        logits = self.temperal_pooling(logits)#[b,c,h,w]
        return logits

    def forward(self, x):
        """[b,c,t,h,w]"""
        #[b,c,t,h,w]->[b,t,c,h,w]
        x0 = x.permute(0,2,1,3,4)
        #[b,t,c,h,w]->[bt,c,h,w]
        x = x0.reshape( tuple([-1]) + x0.shape[-3:])
        return self._forward_mlu_(x)


class UNet2D(BasicModule):
    def __init__(self, n_channels, n_classes):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.sigmoid = nn.Sigmoid()

    def _forward_mlu_(self,x):
        """[b,c,h,w]"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[b,c,h,w]"""
        #[b,c,h,w] -> [b,n_classes,h,w]
        return self._forward_mlu_(x)

    def inference(self, x):
        return self.sigmoid(self._forward_mlu_(x))



class UNet2DMLU(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2DMLU, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.sigmoid = nn.Sigmoid()

    def _forward_mlu_(self,x):
        """[b,c,h,w]"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) #[b,c,h,w]
        return logits

    def forward(self, x):
        """[b,c,h,w]"""
        #[b,c,h,w] -> [b,n_classes,h,w]
        return self.sigmoid(self._forward_mlu_(x))


class UNet3D(BasicModule):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3d(n_channels, 32)
        self.down1 = Down3d(32, 64)
        self.down2 = Down3d(64, 128)
        self.down3 = Down3d(128, 256)
        self.down4 = Down3d(256, 256)
        self.up1 = Up3d(512, 128, bilinear)
        self.up2 = Up3d(256, 64, bilinear)
        self.up3 = Up3d(128, 32, bilinear)
        self.up4 = Up3d(64, 32, bilinear)
        self.outc = OutConv3d(32, n_classes)

    def temperal_pooling(self, x, method='mean'):
        """
        :param x: [b,c,T,h,w] 
        :return: [b,c,h,w] 
        """
        method_pool = ['mean','max']
        assert method in method_pool, 'temperal pooling method should be in {}'.format(method_pool)
        if method == 'mean':
            return torch.mean(x, dim=2)
        elif method == 'max':
            return torch.max(x,dim=2)
        else:
            return torch.mean(x,dim=2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) #[b,c,t,h,w]
        logits = self.temperal_pooling(logits)#[b,c,h,w]
        return logits