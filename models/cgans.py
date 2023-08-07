import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.conv2_1 = nn.Conv2d(2, 1, 1, 1, 0)
        self.conv3_1 = nn.Conv2d(3, 1, 1, 1, 0)
        self.model = nn.Sequential(
            *downsample(1, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            *downsample(512, 512),
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 512),
            *upsample(512, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):

        x1 = self.model(x)

        x2 = self.model(self.conv2_1(torch.cat((x, x1), dim=1)))

        x3 = self.model(self.conv3_1(torch.cat((x, x1, x2), dim=1)))
        return x3


class Discriminator(nn.Module):
    def __init__(self, channels=2):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(channels, 2, False), (64, 2, True), (128, 2, True),
                                               (256, 2, True), (512, 2, True), (512, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 4, 2, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.model(x)