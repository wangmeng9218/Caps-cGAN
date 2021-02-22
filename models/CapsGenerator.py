from models.MultiScaleCaps import *
class CapsBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, vec_length = 16):
        super(CapsBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        in_capsules = inplanes//vec_length
        out_capsules = planes //vec_length
        self.downsample = downsample

        self.conv1 = MultiConvCaps(in_capsules=in_capsules, out_capsules=out_capsules,
                                          in_channels=vec_length,
                                          out_channels=vec_length,
                                          stride=stride, padding=1, kernel=3,
                                          nonlinearity="sqaush", batch_norm=True,
                                          num_routes=3, cuda=True)
        self.conv2 = MultiConvCaps(in_capsules=out_capsules, out_capsules=out_capsules,
                                          in_channels=vec_length,
                                          out_channels=vec_length,
                                          stride=1, padding=1, kernel=3,
                                          nonlinearity="sqaush", batch_norm=True,
                                          num_routes=3, cuda=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out



class CapsGenerator(nn.Module):

    def __init__(self, inchannels=1,block=CapsBlock, layers=[2, 2, 2, 2],
                 num_classes=1,
                 groups=1, width_per_group=64,
                 norm_layer=None,vec_length=16,opt=None):
        super(CapsGenerator, self).__init__()
        self.num_classes = num_classes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.vec_length = vec_length
        self.opt = opt

        self.inplanes = 32
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inchannels, self.inplanes, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0],stride=2,vec_length=self.vec_length)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,vec_length=self.vec_length)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,vec_length=self.vec_length)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,vec_length=self.vec_length)

        self.filters = [32,32,64,128,256]
        self.deconvcaps4 = deconvolutionalCapsule(
            in_capsules=self.filters[-1]//vec_length,
            out_capsules=self.filters[-2]//vec_length,
            in_channels=vec_length,
            out_channels=vec_length,
            stride=2, padding=1, kernel=4,
            nonlinearity="sqaush", batch_norm=True,
            num_routes=3, cuda=True)
        self.deconvcaps3 = deconvolutionalCapsule(
            in_capsules=self.filters[-2]//vec_length,
            out_capsules=self.filters[-3]//vec_length,
            in_channels=vec_length,
            out_channels=vec_length,
            stride=2, padding=1, kernel=4,
            nonlinearity="sqaush", batch_norm=True,
            num_routes=3, cuda=True)

        self.deconvcaps2 = deconvolutionalCapsule(
            in_capsules=self.filters[-3]//vec_length,
            out_capsules=self.filters[-4]//vec_length,
            in_channels=vec_length,
            out_channels=vec_length,
            stride=2, padding=1, kernel=4,
            nonlinearity="sqaush", batch_norm=True,
            num_routes=3, cuda=True)
        self.deconvcaps1 = deconvolutionalCapsule(
            in_capsules=self.filters[-4]//vec_length,
            out_capsules=self.filters[-5]//vec_length,
            in_channels=vec_length,
            out_channels=vec_length,
            stride=2, padding=1, kernel=4,
            nonlinearity="sqaush", batch_norm=True,
            num_routes=3, cuda=True)

        self.convcapse_final = MultiConvCaps(
            in_capsules=self.filters[-5]//vec_length,
            out_capsules=num_classes,
            in_channels=vec_length,
            out_channels=vec_length,
            stride=1, padding=2, kernel=5,
            nonlinearity="sqaush", batch_norm=True,
            num_routes=3, cuda=True)


        self.final_layer = nn.Conv2d(in_channels=self.vec_length, out_channels=1, kernel_size=1, padding=0, stride=1)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.drop4 = nn.Dropout(0.5)
        self.drope1 = nn.Dropout(0.5)
        self.drope2 = nn.Dropout(0.5)
        self.drope3 = nn.Dropout(0.5)
        self.drope4 = nn.Dropout(0.5)


    def _make_layer(self, block, planes, blocks, stride=1,vec_length=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MultiConvCaps(
                    in_capsules=self.inplanes//vec_length,
                    out_capsules=(planes * block.expansion)//vec_length,
                    in_channels=vec_length,
                    out_channels=vec_length,
                    stride=2, padding=2, kernel=5,
                    nonlinearity="sqaush", batch_norm=True,
                    num_routes=3, cuda=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width,vec_length=vec_length))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, vec_length=vec_length))

        return nn.Sequential(*layers)

    def forward(self, x, isTrain=True):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_conv1 = x.reshape(batch_size, self.filters[0]//self.vec_length,
                            self.vec_length, x.size(-2), x.size(-1))


        x_1 = self.layer1(x_conv1)
        if isTrain:
            x_1 = self.drop1(x_1)

        x_2 = self.layer2(x_1)
        if isTrain:
            x_2 = self.drop2(x_2)

        x_3 = self.layer3(x_2)
        if isTrain:
            x_3 = self.drop3(x_3)

        x_4 = self.layer4(x_3)
        if isTrain:
            x_4 = self.drop4(x_4)

        d_4 = self.deconvcaps4(x_4)+x_3
        if isTrain:
            d_4 = self.drope1(d_4)

        d_3 = self.deconvcaps3(d_4)+x_2
        if isTrain:
            d_3 = self.drope2(d_3)

        d_2 = self.deconvcaps2(d_3)+x_1
        if isTrain:
            d_2 = self.drope3(d_2)

        d_1 = self.deconvcaps1(d_2)+x_conv1
        if isTrain:
            d_1 = self.drope4(d_1)

        out = self.convcapse_final(d_1)
        x_out = out.view(batch_size, self.vec_length, out.size(-2), out.size(-1))
        out = torch.tanh(self.final_layer(x_out))
        return out


if __name__ == "__main__":
    import torch
    import numpy as np
    print('begin...')
    opt = {}
    opt["isTrain"] = True

    # model = Encoder_Path(64)
    model = CapsGenerator(num_classes=1,opt=opt).cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    tmp = torch.randn(4, 1, 256, 512).cuda()

    print(model(tmp)[0].shape)
    print(model(tmp)[1].shape)
    # print(model(tmp).shape)
    # print('done')



