import numpy as np
import os
import torch
import torch.nn as nn
import pickle
from BaseModel import BaseSeqModel
from utils import PointWiseFeedForward

class SASRec(BaseSeqModel):
    def __init__(self, user_num, item_num, device, args):
        super(SASRec, self).__init__(user_num, item_num, device, args)

        # Embedding层
        self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Transformer结构
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        # 构建多层Transformer
        for _ in range(args.trm_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_size,
                                                        args.num_heads,
                                                        args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self._init_weights()

    def transformer_forward(self, seqs, log_seqs):
        """Transformer前向传播"""
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def _get_embedding(self, log_seqs):
        """获取嵌入表示"""
        item_seq_emb = self.item_emb(log_seqs)
        return item_seq_emb

    def log2feats(self, log_seqs, positions):
        """将序列转换为特征"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.transformer_forward(seqs, log_seqs)
        return log_feats

    def forward(self, seq, pos, neg, positions, loss_type='point'): 
        """前向传播，支持不同损失类型"""
        if loss_type == 'seq':  # 序列到序列损失
            return self._forward_seq_loss(seq, pos, neg, positions)
        else:  # 点损失（默认）
            return self._forward_point_loss(seq, pos, neg, positions)

    def _forward_point_loss(self, seq, pos, neg, positions):
        """点损失计算"""
        log_feats = self.log2feats(seq, positions)
        log_feats = log_feats[:, -1, :].unsqueeze(1)

        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss

    def _forward_seq_loss(self, seq, pos, neg, positions):
        """序列损失计算"""
        log_feats = self.log2feats(seq, positions)
        pos_embs = self._get_embedding(pos)
        neg_embs = self._get_embedding(neg)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss

    def predict(self, seq, item_indices, positions, **kwargs):
        """预测函数"""
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_user_emb(self, seq, positions, **kwargs):
        """获取用户嵌入"""
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        return final_feat
