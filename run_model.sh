#!/bin/bash

if [ "$1" == 'discoGAN' ]; then
	python train.py --model discoGAN -d redbubble -n all --batch_size 32 --print_freq 10 --save_model_freq 20000 --save_sample_freq 1000 --save_individual_sample_freq 1000 --which_model_netG unet_256 --n_cpu 16 --gpu_ids 0 --img_size 256 --niter 50
elif [ "$1" == "cycleGAN" ] ; then
    python train.py --model cycleGAN -d redbubble -n all --batch_size 32 --print_freq 10 --save_model_freq 20000 --save_sample_freq 1000 --save_individual_sample_freq 1000 --which_model_netG resnet_9blocks --n_cpu 16 --gpu_ids 0 --img_size 256 --niter 50 --pool_size 50 --no_dropout
elif [ "$1" == "pix2pix" ]; then
	python train.py --model pix2pixGAN -d redbubble -n all --batch_size 32 --print_freq 10 --save_model_freq 20000 --save_sample_freq 1000 --save_individual_sample_freq 1000 --which_model_netG unet_256 --n_cpu 16 --gpu_ids 0 --img_size 256 --niter 50 --which_direction BtoA --lambda_L1 100 --no_lsgan --norm batch --pool_size 0
elif [ "$1" == "models_comp" ]; then
	python test.py --model discoGAN --dataset redbubble --location_model_dir analysis/results --create_models_overview --type InputTarget --which_direction AtoB --comparsion_models cycleGAN pix2pixGAN --save_fig
elif [ "$1" == "model_vis" ]; then
    python test.py --model discoGAN --location_model_dir analysis/results --dataset redbubble --create_model_training_image --save_fig
else
	echo "$1 option not listed."
	echo "Options include:"
	echo "- discoGAN"
	echo "- cycleGAN"
	echo "- pix2pix"
	echo "- model_vis"
	echo "- models_comp"
fi
