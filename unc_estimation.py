r""" This module implements Binary Segmentation Uncertainty Estimation in PyTorch.
Implementation of formulas are inspired by the following implementation:
https://gitlab.com/wdeback/dl-keras-tutorial/blob/master/notebooks/3-cnn-segment-retina-uncertainty.ipynb
"""

import ttach as tta
import numpy as np
import torch
from typing import Tuple, List


def calc_aleatoric(p_hat: np.ndarray) -> np.ndarray:
    return np.mean(p_hat * (1 - p_hat), axis=0)


def calc_epistemic(p_hat: np.ndarray) -> np.ndarray:
    return np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2


def SUE_MCDO(model, batch: torch.tensor, T: int, last_layer: bool) -> Tuple[np.ndarray, np.ndarray]:
    r"""Interface of Binary Segmentation Uncertainty Estimation with Monte Carlo Dropout (MCDO) method for 1 2D slice.
            Inputs supposed to be in range [0, data_range].
            Args:
                model: Trained model with DO turned on during test, e.g. F.dropout2d(x, p, training=True)).
                batch: Tensor with shape (1, C, H, W).
                T: Number of times to run model at test time
                last_layer: Flag whether there is Sigmoid as a last NN layer
            Returns:
                Aleatoric and epistemic uncertainty maps with shapes equal to batch shape
     """
    model.eval()
    predicted = []
    for t in range(T):
        logits = model(batch).cuda()
        prediction = torch.sigmoid(logits).cpu().detach().numpy() if last_layer else logits.cpu().detach().numpy()
        predicted.append(prediction)
    p_hat = np.array(predicted)
    aleatoric = calc_aleatoric(p_hat)
    epistemic = calc_epistemic(p_hat)

    return aleatoric, epistemic


def SUE_TTA(model, batch: torch.tensor, last_layer: bool) -> Tuple[np.ndarray, np.ndarray]:
    r"""Interface of Binary Segmentation Uncertainty Estimation with Test-Time Augmentations (TTA) method for 1 2D slice.
            Inputs supposed to be in range [0, data_range].
            Args:
                model: Trained model.
                batch: Tensor with shape (1, C, H, W).
                last_layer: Flag whether there is Sigmoid as a last NN layer
            Returns:
                Aleatoric and epistemic uncertainty maps with shapes equal to batch shape
     """
    model.eval()
    transforms = tta.Compose(
        [
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    predicted = []
    for transformer in transforms:
        augmented_image = transformer.augment_image(batch)
        model_output = model(augmented_image)
        deaug_mask = transformer.deaugment_mask(model_output)
        prediction = torch.sigmoid(
            deaug_mask).cpu().detach().numpy() if last_layer else deaug_mask.cpu().detach().numpy()
        predicted.append(prediction)

    p_hat = np.array(predicted)
    aleatoric = calc_aleatoric(p_hat)
    epistemic = calc_epistemic(p_hat)

    return aleatoric, epistemic


def SUE_Ensemble(models: List, batch: torch.tensor, last_layer: bool) -> Tuple[np.ndarray, np.ndarray]:
    r"""Interface of Binary Segmentation Uncertainty Estimation with Ensembles method for 1 2D slice.
            Inputs supposed to be in range [0, data_range].
            Args:
                models: Ensemble of trained models.
                batch: Tensor with shape (1, C, H, W).
                last_layer: Flag whether there is Sigmoid as a last NN layer
            Returns:
                Aleatoric and epistemic uncertainty maps with shapes equal to batch shape
     """
    predicted = []
    for model in models:
        model.eval()
        logits = model(batch).cuda()
        prediction = torch.sigmoid(logits).cpu().detach().numpy() if last_layer else logits.cpu().detach().numpy()
        predicted.append(prediction)

    p_hat = np.array(predicted)
    aleatoric = calc_aleatoric(p_hat)
    epistemic = calc_epistemic(p_hat)
    return aleatoric, epistemic

