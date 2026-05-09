import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import cast

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, use_l2_norm=False, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
        # 用于 EMA 追踪长期使用率
        self.register_buffer('usage_count', torch.ones(self.num_embeddings)) 
        self.decay = decay 

    @staticmethod
    def _is_distributed():
        return dist.is_available() and dist.is_initialized()

    def forward(self, inputs: torch.Tensor, run_revive: bool = False):
        usage_count = cast(torch.Tensor, self.usage_count)
        if self.use_l2_norm:
            flat_inputs = F.normalize(inputs, p=2, dim=1)
            codebook = F.normalize(self.embedding.weight, p=2, dim=1)
        else:
            flat_inputs = inputs
            codebook = self.embedding.weight

        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(codebook**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, codebook.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # ==========================================
        # 监控模块：计算 Codebook 健康指标
        # ==========================================
        metrics = {}
        if self.training:
            # 1. 更新 EMA 使用频率
            current_counts = torch.bincount(encoding_indices.flatten(), minlength=self.num_embeddings).float()
            if self._is_distributed():
                # Aggregate token usage over all ranks so EMA tracks global codebook health.
                dist.all_reduce(current_counts, op=dist.ReduceOp.SUM)
            usage_count.data.mul_(self.decay).add_(current_counts, alpha=1 - self.decay)
            
            # 2. 计算当前 Batch 的活跃 Code 数量
            if self._is_distributed():
                metrics["batch_active_codes"] = int((current_counts > 0).sum().item())
            else:
                metrics["batch_active_codes"] = torch.unique(encoding_indices).size(0)
            
            # 3. 计算长期死码数量 (EMA count 极低的 code)
            metrics["ema_dead_codes"] = (usage_count < 0.5).sum().item()
            
            # 4. 计算 Codebook Perplexity (困惑度)
            # 困惑度越高，说明各个 Code 被使用的分布越均匀；最大值为 num_embeddings
            if self._is_distributed():
                total_count = torch.tensor(
                    [encoding_indices.size(0)],
                    device=inputs.device,
                    dtype=current_counts.dtype,
                )
                dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
                avg_probs = current_counts / total_count.clamp_min(1.0)
            else:
                encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
                encodings.scatter_(1, encoding_indices, 1)
                avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            metrics["perplexity"] = perplexity.item()

        # 获取量化向量 & 计算 Loss & STE
        quantized = self.embedding(encoding_indices).squeeze(1)
        if self.use_l2_norm:
            quantized = F.normalize(quantized, p=2, dim=1)
            
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()

        if run_revive:
            revived_count = self.revive_dead_codes(inputs)
            metrics["revived_count"] = revived_count
        # 注意这里多返回了一个 metrics 字典
        return quantized, loss, encoding_indices.squeeze(1), metrics

    # ==========================================
    # 独立出的复活模块：由外部 Training Loop 定期调用
    # ==========================================
    @torch.no_grad()
    def revive_dead_codes(self, inputs: torch.Tensor):
        """定期拉取当前 batch 的特征点，覆盖掉死码"""
        # In distributed mode, only rank 0 mutates codebook weights,
        # then broadcasts updated weights/buffers to keep all replicas in sync.
        usage_count = cast(torch.Tensor, self.usage_count)
        do_revival = True
        if self._is_distributed():
            do_revival = dist.get_rank() == 0

        if not self.use_l2_norm:
            flat_inputs = inputs
        else:
            flat_inputs = F.normalize(inputs, p=2, dim=1)
            
        dead_indices = torch.nonzero(usage_count < 0.5).squeeze(1)
        
        revived_count = int(len(dead_indices))
        if do_revival and revived_count > 0:
            # 从当前输入中随机采样来替换死码
            random_indices = torch.randint(0, flat_inputs.size(0), (len(dead_indices),))
            sampled_inputs = flat_inputs[random_indices]
            
            self.embedding.weight.data[dead_indices] = sampled_inputs
            usage_count.data[dead_indices] = 1.0 # 满血复活

        if self._is_distributed():
            dist.broadcast(self.embedding.weight.data, src=0)
            dist.broadcast(usage_count.data, src=0)
            revived_tensor = torch.tensor([revived_count], device=inputs.device, dtype=torch.int64)
            dist.broadcast(revived_tensor, src=0)
            revived_count = int(revived_tensor.item())

        return revived_count # 返回复活的数量，方便日志记录