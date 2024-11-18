from model_attn import FFN, SelfAttn, InrAttn
from model_tools import SubLayerConnect, clones
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class GeoEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(GeoEncoderLayer, self).__init__()
        self.attn_layer = SelfAttn(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x):
        # (b ,n, l, d)
        x = self.sublayer[0](x, lambda x: self.attn_layer(x, x, x, None))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class GeoEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(GeoEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-2)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(EncoderLayer, self).__init__()
        self.inr_sa_layer = InrAttn(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)
        self.w_q = nn.Linear(features, features)
        self.w_k = nn.Linear(features, features)
        self.w_v = nn.Linear(features, features)
        self.elu = nn.ELU()

    def forward(self, x, str_mat, attn_mask):
        Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)
        x = self.sublayer[0](x, lambda x: self.inr_sa_layer(Q, K, V, str_mat, attn_mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class SASEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(SASEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, str_mat, attn_mask):
        for layer in self.layers:
            x = layer(x, str_mat, attn_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(DecoderLayer, self).__init__()
        self.inr_sa_layer = InrAttn(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x, mem, str_mat, mem_pad_mask):
        x = self.sublayer[0](x, lambda x: self.inr_sa_layer(x, mem, mem, str_mat, mem_pad_mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class Decoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(Decoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mem, str_mat, mem_pad_mask):
        for layer in self.layers:
            x = layer(x, mem, str_mat, mem_pad_mask)
        return self.norm(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) #+ res_att  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            #scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
            #scores.masked_fill(attn_mask == 0.0, -1e9)
            scores += attn_mask
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v).squeeze(1)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual)#, res_attn


# class Embedding(nn.Module):
#     def __init__(self, nb_seq, d_Em):
#         super(Embedding, self).__init__()
#         self.nb_seq = nb_seq
#         self.d_Em = d_Em
#         self.pos_embed = nn.Embedding(nb_seq, d_Em)
#         self.norm = nn.LayerNorm(d_Em)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         pos = torch.arange(seq_len, dtype=torch.long).to(x.device)
#         pos = pos.unsqueeze(0).expand(batch_size, seq_len) 
#         embedding = x + self.pos_embed(pos)
#         Emx = self.norm(embedding)
#         return Emx

class Embedding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super(Embedding, self).__init__()
        self.pos_emb_table = nn.Embedding(max_len, d_model)
        self.pos_vector = torch.arange(max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pos_emb = self.pos_emb_table(self.pos_vector[:x.size(1)].unsqueeze(0).repeat(x.size(0), 1).to(x.device))
        x += pos_emb
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)