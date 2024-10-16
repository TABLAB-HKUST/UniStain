import torch
import random
import os
import numpy as np
import json
from random import choice
import PIL
from diffusers import AutoencoderKL, StableDiffusionXLInstructPix2PixPipeline, StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
from drag_pipeline import DragPipeline
from torchvision import transforms
import cv2


def find_neighbor(save_path, file, af_path):
    corr_y, corr_x = file.split('.')[0].split('_')[-1].split('-')
    index = int(file.split('_')[-2])
    corr_y, corr_x = int(corr_x), int(corr_y)
    # print(file, corr_y, corr_x, index)
    img1, img2 = None, None
    af1, af2 = None, None
    step = 512-8
    col_index = 2
    # print(file)
    if corr_x < step and corr_y < step:  
        # the first image
        return None, None, None, None
    
    elif corr_y < step:
        # the first row
        filename = file.replace('_%.2d_%d-%d.jpg'%(index, corr_x, corr_y),'')
        img1 = PIL.Image.open('%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-col_index, corr_x-step, corr_y)).convert('RGB')
        af1 = PIL.Image.open('%s/%s_%.2d_%d-%d.jpg'%(af_path, filename, index-col_index, corr_x-step, corr_y)).convert('RGB')
        # print('1---', '%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-4, corr_x-step, corr_y))
    elif corr_x < step:
        # the first column
        filename = file.replace('_%.2d_%d-%d.jpg'%(index, corr_x, corr_y),'')
        img1 = PIL.Image.open('%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-1, corr_x, corr_y-step)).convert('RGB')
        af1 = PIL.Image.open('%s/%s_%.2d_%d-%d.jpg'%(af_path, filename, index-1, corr_x, corr_y-step)).convert('RGB')
        # print('2***', '%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-1, corr_x, corr_y-step) )
        
    else:
        filename = file.replace('_%.2d_%d-%d.jpg'%(index, corr_x, corr_y),'')
        img1 = PIL.Image.open('%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-col_index, corr_x-step, corr_y)).convert('RGB')
        af1 = PIL.Image.open('%s/%s_%.2d_%d-%d.jpg'%(af_path, filename, index-col_index, corr_x-step, corr_y)).convert('RGB')
        img2 = PIL.Image.open('%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-1, corr_x, corr_y-step)).convert('RGB')
        af2 = PIL.Image.open('%s/%s_%.2d_%d-%d.jpg'%(af_path, filename, index-1, corr_x, corr_y-step)).convert('RGB')
        # print('3+++', '%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-4, corr_x-step, corr_y), '%s/AF512-100000-1.5-2-HE-%s_%.2d_%d-%d.png'%(save_path, filename, index-1, corr_x, corr_y-step))
    return img1, img2, af1, af2

    



model_path = "timbrooks/instruct-pix2pix"
model_name = '202406-multi_organ-1'
step = 100000
lora_path = '../instruct_pix2pix/output/%s/checkpoint-%d/'%(model_name, step)
pipe = DragPipeline.from_pretrained(model_path,
                                                              safety_checker = None,
                                               torch_dtype=torch.float32
                                              )

pipe.modify_transformer_forward()
pipe.unet.load_attn_procs(lora_path)

vae = AutoencoderKL.from_pretrained("timbrooks/instruct-pix2pix", subfolder="vae").to(torch.float16).cuda()
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

generator = torch.Generator("cuda").manual_seed(42)
pipe.to("cuda")
# pipe.safety_checker = lambda images, clip_input: (images, False)


path = '/data1/nfs/shilulin/data/202406-total-thin/512/test/Lung/whole/005-006/test-1024/test-512-8'
prename = 'AF512'
files = os.listdir(path)
files.sort()
files = [file for file in files if 'AF' in file]
save_path = "generated_imgs/%s/%s-%d-Lung-wi_0506/"%(model_name,model_name,step)
os.makedirs("%s/"%(save_path), exist_ok=True)

for file in files[:]:
    # results = os.listdir('/data1/nfs/shilulin/project/2023_vs/multi_stain/diffusers/examples/instruct_pix2pix/generated_imgs/20240219-humanliver-2/20240219-humanliver-2-50000')

    image = PIL.Image.open(os.path.join(path, file))
    # image = PIL.Image.open('/data1/nfs/shilulin/data/202406-total-thin/512/test/Lung/tmp/Human_Lung_AF_H006_P2_D1_S1(NQ)_(2)_10X_V2_AF_Reg_12288-12288_0-1440.jpg')
    # image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((512,512))

    ### prepare references
    img1, img2, af1, af2 = find_neighbor(save_path, file, path)
    if img1 is None:
        ref_img, ref_af = img1, af1
    elif img2 is None:
        ref_img = transform(img1).cuda().unsqueeze(dim=0)
        ref_af = transform(af1).cuda().unsqueeze(dim=0)

    else:
        img1 = transform(img1).cuda().unsqueeze(dim=0)
        img2 = transform(img2).cuda().unsqueeze(dim=0)
        af1 = transform(af1).cuda().unsqueeze(dim=0)
        af2 = transform(af2).cuda().unsqueeze(dim=0)
        ref_img = torch.cat([img1, img2], dim=0)
        ref_af = torch.cat([af1, af2], dim=0)


    # ref_img, ref_af = None, None


    num_inference_steps = 50 #20
    image_guidance_scale = 1.5 #default1.5, similarity with original image, higher value more similar
    guidance_scale = 2 #default7.5 similarity with text prompt , higher value more close
    prompts = ['turn it into the Masson trichrome stained image. this is lung tissue','turn it into Elastic van Gieson stained image. this is lung tissue',
               'turn it into the Hematoxylin and eosin stained image. this is lung tissue']

    labels = ['HE','PAS', 'HE','PASM' ]
    negative_prompt = 'blurry, artifacts, low image quality'
    for prompt, label in zip(prompts[2:3], labels[2:3]):
        edited_image = pipe(prompt,
        image=image, 
        ref_x0=ref_img,
        ref_af=ref_af,
        num_inference_steps=num_inference_steps, 
        image_guidance_scale=image_guidance_scale, 
        guidance_scale=guidance_scale,
        generator=generator,
        safety_checker = None,
        )[0]
        # .images[0]
        
        edited_image.save("%s/%s-%d-%s-%s-%s-%s.png"%(save_path, prename, step, image_guidance_scale, guidance_scale, label, file.split('.')[0],))

