import torch
import torch.nn as nn

class Contrastive():
    def __init__(self, args):
        self.temperature_f = args.temperature_f
        self.temperature_l = args.temperature_l
        self.contrastive_w = args.contrastive_w
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
           
    def mask_correlated_samples(self, N):
            mask = torch.ones((N, N))
            mask = mask.fill_diagonal_(0)
            for i in range(N//2):
                mask[i, N//2 + i] = 0
                mask[N//2 + i, i] = 0
            mask = mask.bool()
            return mask

    def loss(self, q, k):
        B, C, N, _ = q.shape
        q = q.view(B * N * N, C)
        k = k.view(B * N * N, C)

        h = torch.cat((q, k), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        print(sim)
        mask = self.mask_correlated_samples(B * N * N * 2)
        sim_masked = sim[mask].view(-1, 1)

        labels = torch.zeros_like(sim_masked)
        loss = self.criterion(sim_masked, labels)
        loss /= len(sim_masked)
        
        return loss * self.contrastive_w
        # def loss(self, q, k):
    #     B, C, N, _ = q.shape
    #     total = B * N * N * 2
    #     q = q.view(B * N * N, C)
    #     k = k.view(B * N * N, C)
        
    #     h = torch.cat((q, k), dim=0)
    #     sim = torch.matmul(h, h.T) / self.temperature_f
        
    #     sim_i_j = torch.diag(sim, B*N*N)
    #     sim_j_i = torch.diag(sim, -B*N*N)

    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(total, 1)
    #     mask = self.mask_correlated_samples(total)
    #     negative_samples = sim[mask].reshape(total, -1)
    #     labels = torch.zeros(total).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=1)
    #     loss = self.criterion(logits, labels)
    #     loss /= total
    #     return loss * self.contrastive_w
            
    # def loss(self, q , k):
    #     # (bs, dim)
    #     # (Batch, dim, n * n)
    #     B = q.shape[0]
    #     N = B * 2
    #     h = torch.cat((q, k), dim=0)
    #     sim = torch.matmul(h, h.T) / self.temperature_f
    #     sim_i_j = torch.diag(sim, B)
    #     sim_j_i = torch.diag(sim, -B)
    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    #     mask = self.mask_correlated_samples(N)
    #     negative_samples = sim[mask].reshape(N, -1)
    #     labels = torch.zeros(N).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=1)
    #     loss = self.criterion(logits, labels)
    #     loss /= N
    #     return loss * self.contrastive_w

