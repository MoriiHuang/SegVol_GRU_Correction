import torch
import torch.nn as nn
import time

# 定义 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return out

# 设置参数
input_size = 10
seq_len = 20
batch_size = 32
hidden_sizes = [32, 64, 128, 256, 512, 1024]  # 不同的 feature size
hidden_size = 512  # 默认的 feature size
input_sizes = [10, 100, 1000,10000]  # 不同的输入大小

# 测试计算速度
for input_size in input_sizes:
    model = GRUModel(input_size, hidden_size)
    model.to('cuda')  # 将模型移动到 GPU
    x = torch.randn(batch_size, seq_len, input_size).to('cuda')  # 随机输入数据
    print(x.shape)
    
    # 预热（避免初始化的影响）
    for _ in range(10):
        _ = model(x)
    
    # 计算时间
    start_time = time.time()
    for _ in range(100):
        _ = model(x)
    elapsed_time = time.time() - start_time
    
    print(f"Feature size: {hidden_size}, Time: {elapsed_time:.4f} seconds")