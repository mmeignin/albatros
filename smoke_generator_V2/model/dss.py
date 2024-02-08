"""
Implementation of the https://arxiv.org/abs/1809.00774 Deep Smoke Segmentation
"""
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch



class dss(nn.Module):
    # All Model Class overwrite the nn.Module class
    def __init__(self, model):
        super(dss, self).__init__()
        self.conv1 = nn.Sequential(*list(model.children())[0][0 : 3])
        #First skip, connect with decodey2
        self.conv2 = nn.Sequential(*list(model.children())[0][4 : 8])
        #Second Skip, connect with decodey1
        self.conv3 = nn.Sequential(*list(model.children())[0][9 : 15])
        # Third Skip, connect with decodex2
        self.conv4 = nn.Sequential(*list(model.children())[0][16 : 22])
        #Forth Skip, connect with decodex1
        self.conv5 = nn.Sequential(*list(model.children())[0][23 : 28])
        # The twopath structure are named with x and y, where x refer to first path and y refer to second path
        self.decodey1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.decodey2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.decodey3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decodex1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        skip1 = self.conv1(x)
        skip2 = self.conv2(skip1)
        skip3 = self.conv3(skip2)
        skip4 = self.conv4(skip3)
        out = self.conv5(skip4)
        x = self.decodex1(out)
        x = torch.cat((x, skip4), dim=1)
        x = self.decodex2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.decodex3(x)
        y = self.decodey1(skip3)
        y = torch.cat((y, skip2), dim=1)
        y = self.decodey2(y)
        y = torch.cat((y, skip1), dim=1)
        y = self.decodey3(y)
        x = torch.add(x,y)
        x = self.decode(x)
        return x

def test():
    x = torch.randn((3, 1, 256, 256))
    model = dss(in_channels=3, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()

