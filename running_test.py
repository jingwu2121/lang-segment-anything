'''
    This is just adapted from the example in the readme,
    The main usage is for the built image to have the weights cached.
'''
import os
from PIL import Image
from lang_sam import LangSAM
import numpy as np

model = LangSAM()

image_dir = 'test_data/garden_512/rgb'
save_dir = 'test_data/garden_512'
imgfiles = sorted(os.listdir(image_dir)) 
for imgfile in imgfiles:
    imgpath = os.path.join(image_dir, imgfile) 
    
    image_pil = Image.open(imgpath).convert("RGB")
    text_prompt = "table" # "dinosaur statue" # "stone horse" # "bear statue" # "man" # "bear statue"
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

    save_path_npy = os.path.join(save_dir, 'mask_npy', imgfile.replace('jpg', 'npy'))
    os.makedirs(os.path.join(save_dir, 'mask_npy'), exist_ok=True)
    mask_npy_save = masks.clone().cpu().numpy()[0] * 1
    np.save(save_path_npy, mask_npy_save)

    mask_ = masks * 255 
    mask_np = mask_.cpu().numpy()[0] 

    save_path_img = os.path.join(save_dir, 'mask_img', imgfile)
    os.makedirs(os.path.join(save_dir, 'mask_img'), exist_ok=True)
    mask_pil = Image.fromarray(mask_np.astype(np.uint8)) 
    mask_pil.save(save_path_img)

    


print('all ok')