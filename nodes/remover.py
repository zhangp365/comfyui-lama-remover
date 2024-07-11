from PIL import ImageOps, ImageFilter
import torch
from ..utils import cropimage, padimage, padmask, tensor2pil, pil2tensor, cropimage, pil2comfy
from ..lama import model
from torchvision import transforms
import time
import logging
logger = logging.getLogger(__file__)

class LamaRemover:
    def __init__(self) -> None:
        self.mylama = model.BigLama()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold":("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }
    
    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):        
        ten2pil = transforms.ToPILImage()

        results=[]
        
        for image, mask in zip(images, masks):
            ori_image = tensor2pil(image)
            logger.debug(f"input image size :{ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.unsqueeze(0)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            logger.debug(f"input mask size :{ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                logger.debug("resize mask")
                p_mask = p_mask.resize(p_image.size)

            # invert mask
            # 反转遮罩
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)
            
            # gaussian Blur
            # 高斯模糊遮罩（模糊的是白色）
            if gaussblur_radius > 0:
                p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # mask_threshold
            # 遮罩阈值，越大越强
            gray = p_mask.point(lambda x: 0 if x>mask_threshold else 255)

            pt_mask = pil2tensor(gray)

            # lama
            # lama模型
            start = time.time()
            result = self.mylama(pt_image, pt_mask)
            logger.debug(f"lama reference cost:{time.time() - start}")
            
            # 裁剪成输入大小
            _, result_h, result_w = result.size()

            if result_h > h or result_w > w:
                result = result[:, :h, :w]  # 裁剪结果Tensor

            # 转换为ComfyUI格式 (i, h, w, c)
            i = result.permute(1, 2, 0).unsqueeze(0).cpu()  # 将Tensor维度从 (channels, height, width) 变为 (height, width, channels)
            results.append(i)

        return (torch.cat(results, dim=0),)



class LamaRemoverIMG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("IMAGE",),
                "mask_threshold":("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }
    
    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover_IMG"

    def lama_remover_IMG(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        ten2pil = transforms.ToPILImage()

        results=[]
        
        for image, mask in zip(images, masks):
            ori_image = tensor2pil(image)
            logger.debug(f"input image size :{ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.movedim(0, -1).movedim(0,-1)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            logger.debug(f"input mask size :{ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                logger.debug("resize mask")
                p_mask = p_mask.resize(p_image.size)

            # invert mask
            # 反转遮罩
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            # gaussian Blur
            # 高斯模糊遮罩（模糊的是黑色所以需要反转操作）
            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # mask_threshold
            # 遮罩阈值，越大越强
            gray = p_mask.point(lambda x: 0 if x>mask_threshold else 255)

            pt_mask = pil2tensor(gray)

            # lama
            # lama模型
            start = time.time()
            result = self.mylama(pt_image, pt_mask)
            logger.debug(f"lama reference cost:{time.time() - start}")

            img_result = ten2pil(result)

            # crop into the original size
            # 裁剪成输入大小
            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            # turn to comfyui tensor
            # 变成comfyui格式（i,h,w,c）
            i = pil2comfy(img_result)
            results.append(i)
       
        return (torch.cat(results, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "Big lama Remover",
    "LamaRemoverIMG": "Big lama Remover(IMG)"
}