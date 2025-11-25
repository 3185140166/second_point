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
    融合问题embedding和多段passage embedding的交叉注意力模块
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, question_embedding, passage_embeddings):
        """
        Args:
            question_embedding: (batch, 1, hidden_dim)
            passage_embeddings: (batch, num_passages, hidden_dim)

        Returns:
            fused_embedding: (batch, hidden_dim)
        """
        # 交叉注意力 (Q=question, K,V=passages)
        attn_output, attn_weights = self.cross_attention(
            query=question_embedding,         # (batch, 1, hidden_dim)
            key=passage_embeddings,           # (batch, num_passages, hidden_dim)
            value=passage_embeddings          # (batch, num_passages, hidden_dim)
        )
        # 残差连接 + LayerNorm
        x = self.layer_norm1(question_embedding + self.dropout1(attn_output))  # shape (batch, 1, hidden_dim)
        
        # 去掉长度1维度 => (batch, hidden_dim)
        x = x.squeeze(1)
        # MLP融合，残差 + LayerNorm
        mlp_out = self.mlp(x)
        x = self.layer_norm2(x + self.dropout2(mlp_out))

        # x.shape = (batch, hidden_dim)
        return x

