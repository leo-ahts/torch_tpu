import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import matplotlib.pyplot as plt
import pandas as pd
import torch_tpu

# 定义一个模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 初始化
model = SimpleCNN().to('tpu')
inputs = torch.randn(16, 1, 32, 32).to('tpu') # Batch size of 16, 1 channel, 32x32 image
target = torch.randint(0, 10, (16,)).to('tpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for epoch in range(5): # Simulate 5 epochs of training
        optimizer.zero_grad()
        
        with torch.profiler.record_function("TPU_Operation: TPU_FORWARD_PASS"):
            outputs = model(inputs)
        
        with torch.profiler.record_function("TPU_Operation: TPU_LOSS_COMPUTATION"):
            loss = criterion(outputs, target)
        
        with torch.profiler.record_function("TPU_Operation: TPU_BACKWARD_PASS"):
            loss.backward()
        
        with torch.profiler.record_function("TPU_Operation: TPU_OPTIMIZER_STEP"):
            optimizer.step()
        
        prof.step()
# 性能数据
events = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
print(events)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs')
writer.add_text('Profiler Report', str(events))
writer.close()
