import PIL
from PIL import Image, ImageDraw ,ImageFont
from matplotlib import pyplot as plt
import torchvision.transforms as T
import os
import torch 
import yaml
import numpy as np

def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")

def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images

def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]    
    return pil_imgs

def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0)*2-1
    elif type(pil_imgs) == list:    
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0)*2-1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs


def add_margin(pil_img, top = 0, right = 0, bottom = 0, 
                    left = 0, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    
    result.paste(pil_img, (left, top))
    return result
 

from PIL import Image, ImageDraw, ImageFont
import torch

from PIL import Image, ImageDraw, ImageFont
import torch

def image_grid(imgs, rows=1, cols=None, size=None, titles=None, text_pos=(0, 0), binary=False, threshold=128):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)

    if size is not None:
        imgs = [img.resize((size, size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows * cols

    top = 20
    w, h = imgs[0].size
    delta = 0
    if len(imgs) > 1 and imgs[1].size[1] != h:
        delta = top
        h = imgs[1].size[1]
    if titles is not None:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                                    size=20, encoding="unic")
        h = top + h
    grid = Image.new('RGB', size=(cols * w, rows * h + delta))

    for i, img in enumerate(imgs):
        # Convert the image to grayscale if binary is specified
        if binary:
            img = img.convert('L')  # 转换为灰度图像
            # 应用阈值处理
            img = img.point(lambda p: 255 if p > threshold else 0)  # 应用阈值
            
        if titles is not None:
            img = add_margin(img, top=top, bottom=0, left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i], (0, 0, 0),
                       font=font)
        
        if delta != 0 and i > 0:
            grid.paste(img, box=(i % cols * w, i // cols * h + delta))
        else:
            grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

"""
input_folder - dataset folder
"""
def load_dataset(input_folder):
    # full_file_names = glob.glob(input_folder)
    # class_names = [x[0] for x in os.walk(input_folder)]
    class_names = next(os.walk(input_folder))[1]
    class_names[:] = [d for d in class_names if not d[0] == '.']
    file_names=[]
    for class_name in class_names:
        cur_path = os.path.join(input_folder, class_name)
        filenames = next(os.walk(cur_path), (None, None, []))[2]
        filenames = [f for f in filenames if not f[0] == '.']
        file_names.append(filenames)
    return class_names, file_names


def dataset_from_yaml(yaml_location):
    with open(yaml_location, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded

def load_256(image_path, left=0, right=0, top=0, bottom=0, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((256, 256)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    return image

def load_512(image_path, left=0, right=0, top=0, bottom=0, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    return image