import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    优化版交叉注意力融合网络
    适合联合训练和单独训练两种方案，同时支持单文档和多文档处理
    核心设计：固定2层自注意力+1层交叉注意力，增强文档权重预测
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, mlp_ratio=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 固定2层文档自注意力（参考Transformer设计）
        # 第一层文档自注意力
        self.doc_self_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # 第二层文档自注意力
        self.doc_self_attn2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 2. 文档自注意力的LayerNorm和Dropout
        self.doc_norm1 = nn.LayerNorm(hidden_dim)
        self.doc_norm2 = nn.LayerNorm(hidden_dim)
        self.doc_dropout = nn.Dropout(dropout)
        
        # 3. 核心交叉注意力层（问题引导的文档融合）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_dropout = nn.Dropout(dropout)
        
        # 4. 增强版文档权重预测（简化结构，提高效率）
        self.doc_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        
        # 5. 优化版MLP（参考Transformer前馈网络设计）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim)
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp_dropout = nn.Dropout(dropout)
        

    def forward(self, question_embedding, passage_embeddings):
        """
        Args:
            question_embedding: (batch, 1, hidden_dim)
            passage_embeddings: (batch, num_passages, hidden_dim)
        
        Returns:
            fused_embedding: (batch, hidden_dim)
        """
        # batch_size, num_passages, hidden_dim = passage_embeddings.shape
        
        # 多文档情况：固定2层自注意力+1层交叉注意力
        
        # 1. 第一层文档自注意力
        doc_attn_out1, _ = self.doc_self_attn1(
            passage_embeddings, passage_embeddings, passage_embeddings
        )
        passages_enhanced1 = self.doc_norm1(passage_embeddings + self.doc_dropout(doc_attn_out1))
        
        # 2. 第二层文档自注意力
        doc_attn_out2, _ = self.doc_self_attn2(
            passages_enhanced1, passages_enhanced1, passages_enhanced1
        )
        passages_enhanced = self.doc_norm2(passages_enhanced1 + self.doc_dropout(doc_attn_out2))
        
        # 3. 文档权重预测：动态学习文档重要性
        doc_weights = self.doc_weight(passages_enhanced)  # (batch, num_passages, 1)
        weighted_passages = passages_enhanced * doc_weights  # (batch, num_passages, hidden_dim)
        
        # 4. 交叉注意力：问题引导的文档融合
        cross_attn_out, cross_attn_weights = self.cross_attn(
            query=question_embedding,  # (batch, 1, hidden_dim)
            key=weighted_passages,     # (batch, num_passages, hidden_dim)
            value=weighted_passages    # (batch, num_passages, hidden_dim)
        )
        
        # 5. 残差连接 + LayerNorm
        question_enhanced = self.cross_norm(question_embedding + self.cross_dropout(cross_attn_out))
        
        # 6. 去掉长度维度 => (batch, hidden_dim)
        fused = question_enhanced.squeeze(1)
        
        # 7. MLP融合 + 残差连接
        mlp_out = self.mlp(fused)
        fused = self.mlp_norm(fused + self.mlp_dropout(mlp_out))
        
        return fused, doc_weights, cross_attn_weights
