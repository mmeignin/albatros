"""
Implementation of the https://arxiv.org/abs/1809.00774 Deep Smoke Segmentation
"""
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch



class dss(nn.Module):
    def __init__(
            self,
    ):
        super(dss, self).__init__()
        


    def forward(self, x):
       
        return 

def test():
    x = torch.randn((3, 1, 256, 256))
    model = dss(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()

