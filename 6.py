import torch
import torch_tpu
import torch.profiler

model = torch.nn.Linear(10, 10).to('tpu')
inputs = torch.randn(5, 10).to('tpu')
target = torch.randn(5, 10).to('tpu')
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,  # 记录 tensor 的形状
    profile_memory=True,  # 记录内存使用
    with_stack=True  # 在操作中记录堆栈信息
) as prof:
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

prof.export_chrome_trace("trace.json")