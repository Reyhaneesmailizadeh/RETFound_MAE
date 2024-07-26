
import argparse
import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Visulization', add_help=False)
    # image path
    parser.add_argument('--image_url', default='',type=str,
                        help='image path')
     # * checkpoint weights
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    
    return parser


def main(args):
    # define the utils

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    def show_image(image, title=''):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
        return

    def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
        return model

    def run_one_image(img, model):
        x = torch.tensor(img)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE
        loss, y, mask = model(x.float(), mask_ratio=0.75)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [24, 24]

        plt.subplot(1, 4, 1)
        show_image(x[0], "original")

        plt.subplot(1, 4, 2)
        show_image(im_masked[0], "masked")

        plt.subplot(1, 4, 3)
        show_image(y[0], "reconstruction")

        plt.subplot(1, 4, 4)
        show_image(im_paste[0], "reconstruction + visible")

        plt.show()
        # This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)


    img_url = args.image_url # fox, from ILSVRC2012_val_00046145
    img = Image.open(img_url)

    # download checkpoint if not exist
    chkpt_dir = args.finetune
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    img = np.array(img)
    print("converted")
    run_one_image(img, model_mae)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
