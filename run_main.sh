#!/usr/bin/env bash
python main.py \
 --image_path '/data/ISBI-2017/ISIC-2017_Training_Data/'  \
 --model_path '/MedicalAI/runsUNBO/ISIC_Train_Noise_Adam_Dice_Curve_96_relabel_Nws_durintTr.pt'  \
 --result_path '/home/eredekop/ISBI/result_unc/'  \
 --unc_method 'TTA' \
 --gpu_number '0'
