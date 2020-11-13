import cv2
import torch
from models.unet_do import UNet_DO
from models.unet import UNet
import numpy as np
from unc_estimation import SUE_Ensemble, SUE_TTA, SUE_MCDO
import argparse
import os
from typing import List
from main_utils import parse, load_image, load_model, to_tensor


def main():

    parser = argparse.ArgumentParser()
    args = parse(parser)
    """Make folder to save results"""
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    """Load image, transform to tensor and add to batch"""
    image_path = args.image_path
    im = to_tensor(load_image(image_path)).cuda()
    model_path = args.model_path

    """If the desired SUE method - Ensemble, user provide folder to ensemble of saved models, which are loaded to 
    list of models """
    if args.unc_method == 'Ensemble':
        model = []
        paths = [k for k in os.listdir(model_path)]
        for _path in paths:
            model.append(load_model(model_path + _path, _n_channels=im.shape[1], _n_classes=1).cuda())
    else:
        model = load_model(model_path, _n_channels=im.shape[1], _n_classes=1).cuda()

    """Calculate uncertainty with the desired method"""
    if args.unc_method == 'TTA':
        aleatoric, epistemic = SUE_TTA(model, batch=im, last_layer=False)
    elif args.unc_method == 'Ensemble':
        aleatoric, epistemic = SUE_Ensemble(model, batch=im, last_layer=False)
    elif args.unc_method == 'MCDO':
        aleatoric, epistemic = SUE_MCDO(model, batch=im, last_layer=False)

    np.save(args.result_path + 'aleatoric.npy', aleatoric)
    np.save(args.result_path + 'epistemic.npy', epistemic)
    print('Done!')


if __name__ == '__main__':
    main()
