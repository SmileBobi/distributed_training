import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
# from fairscale.optim import OffloadOptimizer
from fairscale.optim import OSS

# 安装依赖: pip install torch fairscale

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embed(tgt)
        output = self.decoder(tgt_emb, memory)
        return self.fc_out(output)

# 训练配置
device = torch.device('cuda')
batch_size = 32
seq_len = 50
vocab_size = 1000
d_model = 512

# 生成虚拟数据
tgt = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
memory = torch.randn(seq_len, batch_size, d_model).to(device)

# 初始化模型
model = TransformerDecoderModel(vocab_size, d_model).to(device)

# 使用Fairscale的OffloadOptimizer
base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = OSS(params=base_optimizer.param_groups, optim=torch.optim.Adam, cpu_offload=True)
# optimizer = OffloadOptimizer(
#     base_optimizer,
#     device=device,
#     offload_device=torch.device("cpu"),
#     offload=True
# )

# 训练循环
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output = model(tgt, memory)
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, vocab_size),
        tgt.view(-1)
    )
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")