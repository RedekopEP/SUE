#!/usr/bin/env bash
PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python main.py \
 --image_path '/data/ISBI-2017/ISIC-2017_Training_Data/ISIC_0010061.jpg'  \
 --model_path '/MedicalAI/runsUNBO/ISIC_Train_Noise_Adam_Dice_Curve_96_relabel_Nws_durintTr.pt'  \
 --result_path '/home/eredekop/ISBI/result_unc/'  \
 --unc_method 'TTA' \
