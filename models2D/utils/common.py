import torch.nn as nn
import torch.nn.functional as F
import torch


class _double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super(_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class _inconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, flag_do: bool, do_p: int = 0.25):
        super(_inconv, self).__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.flag_do = flag_do
        self.do_p = do_p

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.flag_do:
            x = F.dropout2d(x, p=self.do_p, training=True)
        return x


class _down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, flag_do: bool, do_p: int = 0.25):
        super(_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            _double_conv(in_ch, out_ch)
        )
        self.flag_do = flag_do
        self.do_p = do_p

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.mpconv(x)
        if self.flag_do:
            x = F.dropout2d(x, p=self.do_p, training=True)
        return x


class _up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, flag_do: bool, do_p: int = 0.25, bilinear: bool = True):
        super(_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = _double_conv(in_ch, out_ch)
        self.flag_do = flag_do
        self.do_p = do_p

    def forward(self, x1:  torch.Tensor, x2:  torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.flag_do:
            x = F.dropout2d(x, p=self.do_p, training=True)
        return x


class _outconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(_outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x
