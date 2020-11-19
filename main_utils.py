import cv2
import torch
import torch.nn as nn
from models2D.unet import UNet
from models2D.unet_do import UNet_DO
import numpy as np


def parse(parser):
    arg = parser.add_argument
    arg('--image_path', type=str)
    arg('--model_path', type=str)
    arg('--result_path', type=str)
    arg('--unc_method', type=str, choices=['TTA', 'Ensemble', 'MCDO'])
    arg('--gpu_number', type=str)
    args = parser.parse_args()
    return args


def load_image(_path: str) -> np.ndarray:
    im = cv2.imread(_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    s = 96
    im = cv2.resize(im, (s, s))
    return im


def load_image_3D_CT(_path: str) -> np.ndarray:
    im = cv2.imread(_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    s = 96
    im = cv2.resize(im, (s, s))
    return im


def to_tensor(im: np.ndarray) -> torch.Tensor:
    if len(im.shape) != 3:
        im = im[:, :, None]
    im = np.array([im.transpose((2, 0, 1))])
    return torch.from_numpy(im).to(torch.float32)


def load_model(_path: str, _n_channels: int, _n_classes: int):
    model = UNet(n_channels=_n_channels,
                 n_classes=_n_classes)
    state = torch.load(str(_path))
    epoch = state['epoch']
    model.load_state_dict(state['model'])
    print('Restored model, epoch {}'.format(epoch))
    return model


def load_model_DO(_path: str, _n_channels: int, _n_classes: int):
    model = UNet_DO(n_channels=_n_channels,
                    n_classes=_n_classes)
    state = torch.load(str(_path))
    epoch = state['epoch']
    model.load_state_dict(state['model'])
    print('Restored model, epoch {}'.format(epoch))
    return model


def model_parallel(model):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model
