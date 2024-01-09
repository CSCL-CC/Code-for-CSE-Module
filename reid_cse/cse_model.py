import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from reid_cse.cse_loss import ContinuousSurfaceEmbeddingLoss


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, relu=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicStage(nn.Module):
    def __init__(self, dim_list, max_pool=False):
        super(BasicStage, self).__init__()
        self.blocks = []
        if max_pool:
            self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        for i in range(len(dim_list) - 1):
            block = BasicBlock(dim_list[i], dim_list[i + 1], kernel_size=3 if i % 2 == 0 else 1, stride=1,
                               padding=1 if i % 2 == 0 else 0)
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.inc = Stem(3, 64,stride=2, padding=1)
        self.down1 = (Down(64, 128))
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 256)

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
        feat = self.outc(x)
        return feat


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.stem = Stem(3, 32, kernel_size=3, stride=2, padding=1)
        self.s1 = BasicStage([32, 64, 32, 64])
        self.s2 = BasicStage([64, 128, 64, 128], max_pool=True)
        self.s3 = BasicStage([128, 208, 128, 208, 128, 208], max_pool=True)
        self.s4 = nn.Sequential(
            BasicStage([208, 256, 208, 256, 208, 256], max_pool=True),
            BasicBlock(256, 256, 3, 1, 1)
        )
        self.s4_2 = nn.Sequential(
            BasicBlock(208, 128, 3, 1, 1),
            BasicBlock(128, 512, 3, 2, 1)
        )
        self.s5 = nn.Sequential(
            BasicBlock(768, 256, 3, 1, 1),
            BasicBlock(256, 256, 3, 1, 1)
        )
        self.fusion_s1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.fusion_s2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1)
        )
        self.fusion_s3 = nn.Sequential(
            nn.Conv2d(208, 64, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )
        self.fusion_s5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=4.0, mode='nearest')
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x4_2 = self.s4_2(x3)
        print('x2', x2.shape)
        print('x3', x3.shape)
        print('x4', x4.shape)
        x5 = self.s5(torch.cat([x4_2, x4], 1))
        print('x5', x5.shape)

        # 8x mix
        x = self.fusion_s1(x1) + self.fusion_s2(x2) + self.fusion_s3(x3) + self.fusion_s5(x5)
        x = self.head(x)
        return x


class CSEPredictor(nn.Module):
    """
    predictor of CSE representation and DensePose coarse segmentation (24 parts / background)
    """
    def __init__(self, dim_in, p_chan, cse_chan):
        super(CSEPredictor, self).__init__()
        self.p_chan = p_chan
        self.cse_chan = cse_chan
        self.decode = BasicBlock(
            dim_in,
            p_chan + cse_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            relu=False
        )

    def interp2d(self, size):
        """
        Args:
            tensor_nchw: shape (N, C, H, W)
        Return:
            tensor of shape (N, C, Hout, Wout) by applying the scale factor to H and W
        """
        return nn.functional.interpolate(
            size, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )

    def forward(self, head_outputs):
        x = F.interpolate(head_outputs, scale_factor=2, mode='nearest')
        x = self.decode(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        p_out, cse_out = torch.split(x, [self.p_chan, self.cse_chan], dim=1)
        cse_out = F.normalize(cse_out, 1)
        return p_out, cse_out


class CSEModel(nn.Module):
    def __init__(self, p_chan, cse_chan, backbone='unet'):
        super(CSEModel, self).__init__()
        if backbone == 'unet':
            self.backbone = UNet()
        else:
            self.backbone = DarkNet()
        self.CSEPredictor = CSEPredictor(dim_in=256, p_chan=p_chan, cse_chan=cse_chan)
        self.loss = ContinuousSurfaceEmbeddingLoss(embedding_dim=cse_chan)

    def forward(self, img, dp_masks_gt=None, dp_x=None, dp_y=None, dp_I=None, dp_U=None, dp_V=None,
                mode='loss'):
        x = self.backbone(img)
        dp_masks_pred, cse_pred = self.CSEPredictor(x)
        if mode == 'feat':
            return dp_masks_pred, cse_pred
        elif mode == 'eval':
            self.loss(
                dp_masks_pred, dp_masks_gt,
                cse_pred, dp_x, dp_y, dp_I, dp_U, dp_V, img, evaluate=True
            )
        elif mode == 'loss':
            return self.loss(
                dp_masks_pred, dp_masks_gt,
                cse_pred, dp_x, dp_y, dp_I, dp_U, dp_V, img
            )

@register_model
def cse_unet64(pretrained=True, **kwargs):
    model = CSEModel(p_chan=1, cse_chan=32, backbone='unet')
    return model

@register_model
def cse_darknet19(pretrain=True, backbone='darknet', **kwargs):
    model = CSEModel(p_chan=15, cse_chan=4, backbone='darknet')
    return model

@register_model
def cse_darknet19_binary(pretrained=True, backbone='darknet', **kwargs):
    model = CSEModel(p_chan=1, cse_chan=32, backbone='darknet')
    return model


@register_model
def cse_darknet19_binary2(pretrained=True, **kwargs):
    model = CSEModel(p_chan=1, cse_chan=64, backbone='darknet')
    return model


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    from reid_cse.cse_joint_db import CSEDataset

    # rgb = torch.zeros([128, 3, 256, 128]).cuda()
    net = timm.create_model('cse_unet64').cuda()


    def collate_fn(batch):
        batch_dict = {}
        list_names = ['dp_x', 'dp_y', 'dp_I', 'dp_U', 'dp_V']
        for k in batch[0].keys():
            if k in list_names:
                batch_dict[k] = [x[k] for x in batch]
            else:
                batch_dict[k] = torch.stack([x[k] for x in batch], 0)
        return batch_dict


    dataloader = torch.utils.data.DataLoader(
        CSEDataset(['./data/reid_cse/DP3D_train.json']), batch_size=4, num_workers=8,
        collate_fn=collate_fn, pin_memory=True, shuffle=False
    )
    for i, batch in enumerate(dataloader):
        # with torch.no_grad():
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to('cuda')
        net(**batch, mode='eval')
