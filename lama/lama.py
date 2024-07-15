import os,sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from ..utils import get_models_path
from comfy.model_management import get_torch_device
from omegaconf import OmegaConf


from saicinpainting.training.trainers import load_checkpoint
import folder_paths
import yaml
import logging
logger = logging.getLogger(__name__)


DEVICE = get_torch_device()

class LamaModel:

    def __init__(self):
        self.device = DEVICE

        train_config_path =folder_paths.get_full_path("inpaint",'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = folder_paths.get_full_path("inpaint", 'best.ckpt')
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)

    def __call__(self, image, mask):
        with torch.no_grad():
            batch = {'image': image,'mask':mask} 
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['mask'] = (batch['mask'] > 0) * 1
            logger.debug(f"image shape:{batch['image'].shape},mask shape:{batch['mask'].shape}")
            # 模型推理
            output = self.model(batch)
            prediction = output['inpainted'].squeeze(0)

            return prediction
