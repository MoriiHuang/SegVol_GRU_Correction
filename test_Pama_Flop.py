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

from torch.profiler import profile, record_function, ProfilerActivity

# 示例输入数据
input_data = torch.randn(1, 32, 224, 224)  
# 开始性能分析
with profile(activities=[ProfilerActivity.FORWARD], record_shapes=True) as prof:
    model(input_data)

# 打印推理过程中的 FLOPs
print(f"Number of FLOPs: {prof.total_average().cpu_time / 1e9} GFLOPs") 