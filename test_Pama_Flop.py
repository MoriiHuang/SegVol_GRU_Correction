import torch
from segvol.segvol_gru_correction import SegVolModel
from transformers import AutoTokenizer, AutoConfig
import os
import glob
import numpy as np

model_dir = './segvol'
clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, test_mode=False)
model = SegVolModel(config)
model.model.text_encoder.tokenizer = clip_tokenizer

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {total_params / 1e6}M")  

import torch.nn as nn

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):  # x 是输入的 dict（模拟 kwargs）
        return self.model(**x)
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch

# 输入模拟
input_data = {
    "image": torch.randn(1, 1, 32, 256, 256),  # 示例：3D图像
    "zoomed_image": torch.randn(1, 1, 32, 256, 256),
    "text_prompt": None,
    "bbox_prompt_group": None,
    "point_prompt_group": None,
    "use_zoom": False,
    "stage":"box",
    "train_organs": None,
    "train_labels":  torch.randn(1, 32, 256, 256),
}

# 包装模型
wrapped_model = WrappedModel(model.eval()) 
flops = FlopCountAnalysis(wrapped_model, (input_data,))
params = parameter_count(wrapped_model)

print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print(f"Params: {params[''] / 1e6:.2f} M")
