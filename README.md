## github
https://github.com/SSinyu/WGAN_VGG

# WGAN_VGG
Implementation of Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss
https://arxiv.org/abs/1708.00961    

## Use
1. train
main.py --dataset mayo --lr 0.0001

2. train resume
main.py --dataset mayo --lr 0.0001 --resume

3. test resume best valid checkpoint
main.py --dataset mayo --mode test --resume_best

4. test resume last valid checkpoint
main.py --dataset mayo --mode test
