# _Study_StableViton
In this repo, I am trying to Deeply understand and analysis of [CVPR2024-StableVITON](https://arxiv.org/abs/2312.01725) (model &amp; paper). And find approaches to mimimalize its model size.

[[The main Project Page](https://rlawjdghek.github.io/StableVITON/)]&nbsp;

![teaser](assets/method_overview2.png)&nbsp;

## TODO List
- [x] ~~Inference code~~
- [x] ~~Release model weights~~
- [ ] Sub models documentation
- [ ] Model pretrained weights extraction from checkpoint 
- [ ] Model size and training specs
- [ ] Explain Paint-By-Example in DM (diffusion models) world 
- [ ] Input to output flow throught model
- [ ] Training Section
- [ ] Special points mentioned in the Paper

## Solution Explanation 

The proposed solution looking to the problem as In-Painting problem. the Inpainting problem is well known in diffusion models as Paint-By-Example (PBE).

## model understanding

the model is consist of 
(1) CLIP Image Encoder
(3) UNET model
(2) ControlNet 
(4) Variational autoencoder (VAE)
(5) conditional Latent diffusion model wrapper (CLDM)

## model parts explanations

### CLIP Image Encoder

### TimeEmbedding UNET mode

### ControlNet

### Variational autoencoder (VAE)

### conditional Latent diffusion model wrapper (CLDM)

## Input to output flow throught model

- The image (3D channel) is first enter throught the Encoder of VAE to be out as 4D image in latent space with compromised size.

- Then it pass throught the UNET model(implemented with time Embedding) so that it can predict the noise (eps).

- The input to TimeEmbeddingUNET is (output of the encoder of the VAE + The out from CLIP image Encoder as **hint** + the Concat of agnostics+densepose as **Condition**) 

- Thought time, the model learn how to remove the noise from the starting point (zt). And replace the mask (agnostic) area with the cloths in a way fitting the pose of the person.


# Training Section
## Download necessary weights for finetuning

(1) download the weights,  The original weights needed for training is existed in this link [checkpoint](https://kaistackr-my.sharepoint.com/:f:/g/personal/rlawjdghek_kaist_ac_kr/EjzAZHJu9MlEoKIxG4tqPr0BM_Ry20NHyNw5Sic2vItxiA?e=5mGa1c).

But to ease downloading process, Weights can be downloaded directly using the following commands

```
download PBE: gdown 'https://drive.google.com/uc?export=download&id=12tk1e4PYKeD9JZ4FAa2uf_r-LfCf7Nsw'


download VAE: gdown 'https://drive.google.com/uc?export=download&id=1cB1SMyn4QX8xQFvXaO98b7sjVefpBfR7'

```
### understand config file



## Prepare Dataset
If you are going to build you own dataset. You can follow [my Repo]() to prepare the dataset for Training.
VITON-HD dataset can be download from [here](https://github.com/shadow2496/VITON-HD).<br>
For both training and inference, the following dataset structure is required:

```
train
|-- image
|-- image-densepose
|-- agnostic
|-- agnostic-mask
|-- cloth
|-- cloth_mask
|-- gt_cloth_warped_mask (for ATV loss)

test
|-- image
|-- image-densepose
|-- agnostic
|-- agnostic-mask
|-- cloth
|-- cloth_mask
```

## Training
For VITON training, we increased the first block of U-Net from 9 to 13 channels (add zero conv) based on the Paint-by-Example (PBE) model. Therefore, you should download the modified checkpoint (named as 'VITONHD_PBE_pose.ckpt') from the [Link](https://kaistackr-my.sharepoint.com/:f:/g/personal/rlawjdghek_kaist_ac_kr/EjzAZHJu9MlEoKIxG4tqPr0BM_Ry20NHyNw5Sic2vItxiA?e=5mGa1c) and place it in the './ckpts/' folder first.

Additionally, for more refined person texture, we utilized a VAE fine-tuned on the VITONHD dataset. You should also download the checkpoint (named as VITONHD_VAE_finetuning.ckpt') from the [Link](https://kaistackr-my.sharepoint.com/:f:/g/personal/rlawjdghek_kaist_ac_kr/EjzAZHJu9MlEoKIxG4tqPr0BM_Ry20NHyNw5Sic2vItxiA?e=5mGa1c) and place it in the './ckpts/' folder.

```bash
### Base model training
CUDA_VISIBLE_DEVICES=3,4 python train.py \
 --config_name VITONHD \
 --transform_size shiftscale3 hflip \
 --transform_color hsv bright_contrast \
 --save_name Base_test

### ATV loss finetuning
CUDA_VISIBLE_DEVICES=5,6 python train.py \
 --config_name VITONHD \
 --transform_size shiftscale3 hflip \
 --transform_color hsv bright_contrast \
 --use_atv_loss \
 --resume_path <first stage model path> \
 --save_name ATVloss_test
```

