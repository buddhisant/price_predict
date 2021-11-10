# 在训练时，每张显卡上计算的样本数量
samples_per_gpu = 64

# 训练的epochs数量
max_epochs = 30

# 学习率
lr=0.01

# 基础weight_decay率
weight_decay=0.0001

# 优化器的动量
momentum=0.9

# 保存checkpoint的路径
archive_path="archive"

# 保存checkpoint的文件名前缀
check_prefix="gru"