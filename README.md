# _Study_StableViton
In this repo, I am trying to Deeply understand and analysis of [CVPR2024-StableVITON](https://arxiv.org/abs/2312.01725) (model &amp; paper). And find approaches to mimimalize its model size.

[[The main Project Page](https://rlawjdghek.github.io/StableVITON/)]&nbsp;

![teaser](assets/method_overview2.png)&nbsp;

## TODO List
- [ ] Solution Explanation
- [ ] Explain Paint-By-Example in DM (diffusion models) world
- [ ] Model understanding
- [ ] Sub Model Understanding
- [ ] Input to output flow throught model
- [ ] Model pretrained weights extraction from checkpoint 
- [ ] Model size and training specsf
- [ ] Training Section
- [ ] Special points mentioned in the Paper

## Solution Explanation 
StableVITON approaches the problem as an inpainting task, which is common in diffusion models and closely related to Paint-By-Example (PBE). The model replaces missing parts (e.g., masked areas) with realistic clothing fitting the pose of the person using a guided approach with agnostic images and densepose.

## Explain Paint-By-Example in DM (diffusion models) world
Paint-By-Example (PBE) refers to using image inpainting techniques, where diffusion models generate or complete an image based on a specific masked region and an example image for guidance. In the Diffusion Model (DM) world, the process involves progressively denoising random noise to reconstruct the missing parts of an image, while ensuring that the missing region is filled according to the example provided. The goal is to maintain coherence with the original image’s context and the condition (example)

## Model understanding

the model is consist of 
(1) CLIP Image Encoder: Provides semantic context from the reference image.
(3) UNET model: Used for the noise prediction during denoising steps.
(2) ControlNet: Guides the generation with structured cues.
(4) Variational autoencoder (VAE): Encodes the input image into a latent representation.
(5) conditional Latent diffusion model wrapper (CLDM): Generates the final output conditioned on the latent inputs.

## Sub Model Understanding

### CLIP Image Encoder
This module extracts semantic features from the input image that guide the generation process.

### TimeEmbedding UNET mode
This UNET model works with temporal embeddings to predict noise during denoising steps, crucial in diffusion-based models.

### ControlNet
ControlNet is used to incorporate structure guidance, ensuring the generated clothing adheres to the target pose and body structure.

### Variational autoencoder (VAE)
The VAE encodes the input image into a lower-dimensional latent space, crucial for dimensionality reduction and compact representation.

### conditional Latent diffusion model wrapper (CLDM)
The CLDM guides the entire diffusion process, conditioned on both the latent representations from the VAE and the structured cues from ControlNet.

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

