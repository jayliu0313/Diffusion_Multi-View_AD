import torch
import torch.nn as nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        # 将特征展平以计算余弦相似度
        x = x.view(x.size(0), -1)
        print(x.shape)
        loss = self.cos(x, x)
        print(loss.shape)
        
        # cosine_sim_matrix.fill_diagonal_(0)
        
        # 计算损失（相似度的均值）
        loss = loss.mean()
        
        return loss