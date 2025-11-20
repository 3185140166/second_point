import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingFusion(nn.Module):
    """
    学习如何融合多个文档的embedding
    支持动态数量的文档
    """
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 自注意力融合
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP融合层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, doc_embeddings):
        """
        Args:
            doc_embeddings: list of tensors, 每个tensor是 (1, hidden_dim) 或 (batch, hidden_dim)
        
        Returns:
            fused_embedding: (batch, hidden_dim) 或 (1, hidden_dim)
        """
        # 堆叠文档embeddings
        if len(doc_embeddings) == 1:
            # 单文档，直接返回
            return doc_embeddings[0]
        
        # 多文档：堆叠
        stacked = torch.cat(doc_embeddings, dim=0)  # (num_docs, hidden_dim)
        batch_size = doc_embeddings[0].shape[0]
        stacked = stacked.reshape(batch_size, -1, self.hidden_dim)  # (batch, num_docs, hidden_dim)
        
        # 自注意力
        attn_output, attn_weights = self.self_attention(
            stacked, stacked, stacked
        )  # (batch, num_docs, hidden_dim)
        
        # 残差连接 + LayerNorm
        attn_output = self.layer_norm1(stacked + self.dropout(attn_output))
        
        # 对所有文档做平均池化
        fused = attn_output.mean(dim=1)  # (batch, hidden_dim)
        
        # MLP
        mlp_output = self.mlp(fused)
        
        # 残差连接 + LayerNorm
        fused = self.layer_norm2(fused + self.dropout(mlp_output))
        
        return fused


class CrossAttentionFusion(nn.Module):
    """
    跨文档注意力融合（适合2-3个文档）
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 对每个文档分别做attention
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(3)  # 支持最多3个文档
        ])
        
        # 最后的融合层
        self.fusion_linear = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, doc_embeddings):
        """
        Args:
            doc_embeddings: list of (batch, hidden_dim) tensors
        
        Returns:
            fused_embedding: (batch, hidden_dim)
        """
        if len(doc_embeddings) == 1:
            return doc_embeddings[0]
        
        # 对每个文档进行交叉注意力
        outputs = []
        for i, doc_emb in enumerate(doc_embeddings):
            # 以当前文档为query，其他所有文档为key/value
            other_docs = torch.cat(
                [doc_embeddings[j] for j in range(len(doc_embeddings)) if j != i],
                dim=0
            )  # (num_other_docs, hidden_dim)
            
            # 扩展维度用于attention
            doc_emb_expanded = doc_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
            other_docs_expanded = other_docs.unsqueeze(0).expand(doc_emb.shape[0], -1, -1)
            
            # 交叉注意力
            attn_out, _ = self.cross_attention_layers[i](
                doc_emb_expanded, 
                other_docs_expanded, 
                other_docs_expanded
            )
            
            outputs.append(attn_out.squeeze(1))  # (batch, hidden_dim)
        
        # 拼接所有输出
        concatenated = torch.cat(outputs, dim=-1)  # (batch, hidden_dim * num_docs)
        
        # 融合
        fused = self.fusion_linear(concatenated)
        fused = self.layer_norm(fused)
        
        return fused
