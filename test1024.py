import torch
torch.manual_seed(0)
print(torch.rand(1)) # 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数