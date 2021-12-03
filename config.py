# 在训练时，每张显卡上计算的样本数量
samples_per_gpu = 128

# 训练的epochs数量
max_epochs = 30

# 学习率
lr=0.0001

# 基础weight_decay率
weight_decay=0.0001

# 优化器的动量
momentum=0.9

# 保存checkpoint的路径
archive_path="archive"

# 保存checkpoint的文件名前缀
check_prefix="cnn"

# 结果放大倍率
scale=1000

# 分类分支，分类预测的初始得分
class_prior_prob=0.01

# focal loss中的gamma
focal_loss_gamma = 2.0
# focal loss中的alpha
focal_loss_alpha = 0.25

# 最大跌幅
max_decline=-0.002
# 最大涨幅
max_increase=0.002
# 分类间隔
interval=0.001
