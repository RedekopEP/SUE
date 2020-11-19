import numpy as np
from unc_estimation import SUE_Ensemble, SUE_TTA, SUE_MCDO
import argparse
import os
import torch
from main_utils import parse, load_image, load_model, to_tensor, load_model_DO, model_parallel
import tqdm
import torch.backends.cudnn as cudnn
import random


def main():
    parser = argparse.ArgumentParser()
    args = parse(parser)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    """Make folder to save results"""
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    """Find out number of channels in images"""
    image_path = args.image_path
    image_path_idxs = [k for k in os.listdir(image_path)]
    im_tmp = load_image(image_path + image_path_idxs[0])
    n_channels = im_tmp.shape[2]

    """If the desired SUE method - Ensemble, user provide folder to ensemble of saved models, which are loaded to 
            list of models """
    model_path = args.model_path
    if args.unc_method == 'Ensemble':
        model = []
        paths = [k for k in os.listdir(model_path)]
        for _path in paths:
            model.append(model_parallel(load_model(model_path + _path, _n_channels=n_channels, _n_classes=1).cuda()))
    elif args.unc_method == 'TTA':
        model = model_parallel(load_model(model_path, _n_channels=n_channels, _n_classes=1).cuda())
    elif args.unc_method == 'MCDO':
        model = model_parallel(load_model_DO(model_path, _n_channels=n_channels, _n_classes=1).cuda())

    tq = tqdm.tqdm(total=(len(image_path_idxs)))
    for _image_path_idxs in image_path_idxs:
        tq.update(1)
        """Load images, transform to tensor and add to batch"""
        im = to_tensor(load_image(image_path + _image_path_idxs))
        im = torch.from_numpy(np.array(im)).cuda()

        """Calculate uncertainty with the desired method"""
        if args.unc_method == 'TTA':
            aleatoric, epistemic = SUE_TTA(model, batch=im, last_layer=False)
        elif args.unc_method == 'Ensemble':
            aleatoric, epistemic = SUE_Ensemble(model, batch=im, last_layer=False)
        elif args.unc_method == 'MCDO':
            aleatoric, epistemic = SUE_MCDO(model, batch=im, last_layer=False, T=10)

        """Save images to result_path"""
        np.save(args.result_path + _image_path_idxs.split('.')[0] + '_aleatoric.npy', aleatoric)
        np.save(args.result_path + _image_path_idxs.split('.')[0] + '_epistemic.npy', epistemic)
    print('Done!')


if __name__ == '__main__':
    main()
