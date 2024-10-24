# --------------------------------------------------------
# FEA-T
# Written by Yantao Chen
#
# This Code present the FEA-T network proposed in 'Abandon locality: Frame-wise Embedding Aided
# Transformer for Automatic Modulation Recognition'. The class FEA_T(nn.Module) is the backbone
# The default hyper-parameters is seted according to Section Ⅳ-A of our paper. Some extension options
# (such as real_former/feature_mode/ffn_type(DB-GLU-cat/cross_add)) are provided for future research.
#
# Syntax
# MyModel = FEA_T().cuda()  # Initialization
# Y_est = MyModel(X.float(), state='Eval')  # # Classification
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from timm.models.layers import trunc_normal_

class MHSA_block(nn.Module):
    # The Multi-head self attention block of the transformer
    # input format: (Batch, Patch, Dim)
    def __init__(self,
                 d_model=64,  # The input dimension (d_model=2L, L is the frame length)
                 d_fix_qk=16,  # The embedding dimension of queries and keys (i.e., d_p in the paper)
                 d_fix_v=16,  # The embedding dimension of value (i.e., d_p in the paper)
                 n_head_qk=4,  # The head number of queries and keys (i.e., h in the paper)
                 n_head_v=4,  # The head number of value (i.e., h in the paper)
                 dropout=0.,  # The dropout ratio for the value and context attention matrix
                 bias=False,  # If implement a learnable bias for all linear projection
                 talking=True,  # If implement the talking-head attention technique
                 attn_res=False):  # If implement the RealFormer technique (We provide this exploration, but not utilized in FEA-T)
        super(MHSA_block, self).__init__()
        self.dm = d_model
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h_qk = n_head_qk
        self.h_v = n_head_v
        self.attn_res = attn_res
        # self.scale = torch.sqrt(torch.tensor(d_fix_qk))
        self.scale = torch.tensor(d_fix_qk) ** 0.5
        self.to_q = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)  # W_q
        self.to_k = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)  # W_k
        self.to_v = nn.Linear(d_model, d_fix_v * n_head_v, bias=bias)  # W_v
        self.proj_bf = nn.Conv2d(n_head_qk, n_head_qk, (1, 1), bias=False) if talking else nn.Identity()  # W_t
        self.proj_v = nn.Linear(d_fix_v*n_head_v, d_model, bias=bias)  # W_out
        self.softmax = nn.Softmax(dim=-1)
        self.Dp_attn = nn.Dropout(dropout)
        self.Dp_v = nn.Dropout(dropout)

    def forward(self, x, attn_=None, return_attn=False):
        # input format: (Batch, Patch, Dim)
        # attn_ is the context attention matrix of last transformer layer
        # return_attn: if return the context attention matrix
        B, P, D = x.shape
        attn_ = attn_ if attn_ is not None else 0.
        # Generate the queries, keys, and values
        q = self.to_q(x).reshape(B, P, self.h_qk, self.df_qk).permute(0, 2, 1, 3) / self.scale  # (B, h, P, df)
        k = self.to_k(x).reshape(B, P, self.h_qk, self.df_qk).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, P, self.h_v, self.df_v).permute(0, 2, 1, 3)
        # Scaled dot-product attention
        attn = q @ k.transpose(-2, -1)  # (B, h, P, P)
        attn = self.proj_bf(attn)
        attn = self.softmax(attn + attn_)
        attn = self.Dp_attn(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, P, self.df_v*self.h_v)
        # Output projection of MHSA
        x = self.Dp_v(self.proj_v(x))
        if return_attn or self.attn_res:
            return x, attn
        return x


class MLP_block(nn.Module):
    # The MLP scheme for FFN block of the transformer
    def __init__(self,
                 d_model=64,  # The input dimension (d_model=2L, L is the frame length)
                 dim_feedforward=256,  # The dimension of inner-layer (i.e., the output dimension of W_fc1)
                 dropout=0.,  # The dropout ratio for inner-layer
                 activate='relu'):
        super(MLP_block, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward,)  # W_fc1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,)  # W_fc2
        if activate == 'relu':
            self.Act = nn.ReLU()
        elif activate == 'elu':
            self.Act = nn.ELU()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        # input format: (Batch, Patch, Dim)
        return self.linear2(self.dropout(self.Act(self.linear1(x))))


class GLU_block(nn.Module):
    # The GLU scheme for FFN block of the transformer
    def __init__(self,
                 d_model=64,  # The input dimension (d_model=2L, L is the frame length)
                 dim_feedforward=128,  # The dimension of inner-layer (i.e., the output dimension of W_fc1)
                 dropout=0.,  # The dropout ratio for inner-layer
                 activate='relu'):
        super(GLU_block, self).__init__()
        self.dim_f = dim_feedforward//2
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward//2, d_model,)
        if activate == 'relu':
            self.Act = nn.ReLU()
        elif activate == 'sigmoid':
            self.Act = nn.Sigmoid()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        # input format: (Batch, Patch, Dim)
        x_ = self.linear1(x)
        x, x_gate = x_[..., :self.dim_f], x_[..., self.dim_f:]
        return self.linear2(self.dropout(x * self.Act(x_gate)))


class DB_GLU_block(nn.Module):
    # The DB-GLU scheme for FFN block of the transformer
    def __init__(self,
                 d_model=64,  # The input dimension (d_model=2L, L is the frame length)
                 dim_feedforward=128,  # The dimension of inner-layer (i.e., the output dimension of W_fc1)
                 dropout=0.,  # The dropout ratio for inner-layer
                 activate='relu',
                 merge_method='add'):  # We present 3 merge methods for two branches ('add' mode perform eq. (6) of the paper)
        super(DB_GLU_block, self).__init__()
        self.merge_method = merge_method
        self.dim_f = dim_feedforward//2
        dim_linear2 = dim_feedforward//2 if merge_method != 'cat' else dim_feedforward
        if merge_method == 'cross_add' and dropout == 0.:
            dropout = 0.1
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout) if merge_method == 'cross_add' else nn.Identity()
        self.linear2 = nn.Linear(dim_linear2, d_model,)
        if activate == 'relu':
            self.Act = nn.ReLU()
        elif activate == 'sigmoid':
            self.Act = nn.Sigmoid()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        # input format: (Batch, Patch, Dim)
        x_ = self.linear1(x)
        x_1, x_2 = x_[..., :self.dim_f], x_[..., self.dim_f:]
        if self.merge_method == 'add':
            return self.linear2(self.dropout(x_1 * self.Act(x_2) + x_2 * self.Act(x_1)))
        elif self.merge_method == 'cross_add':
            return self.linear2(self.dropout(x_1 * self.Act(x_2)) + self.dropout_2(x_2 * self.Act(x_1)))
        else:
            return self.linear2(self.dropout(torch.cat((x_1 * self.Act(x_2), x_2 * self.Act(x_1)), dim=-1)))


class Frame(nn.Module):  # FEM module
    def __init__(self,
                 PatchSize=32,  # Frame length
                 n_patch=32,  # Frame number
                 overlap=0.5,  # Sliding step length.
                 bias=False):
        super(Frame, self).__init__()
        self.Stride=int(PatchSize*overlap)
        self.PatchSize = PatchSize
        self.FrameNum = n_patch - 1
        # self.Embeddinbg = nn.Conv1d(2, PatchSize*2, (PatchSize,), stride=(self.Stride,), bias=bias)
        self.Embeddinbg = nn.Linear(PatchSize*2, PatchSize*2, )


    def forward(self, x):
        # input_format: (B, 1024, 2)
        # Note that r_I and r_Q are concatenate in the last dimension of x
        Input_Feature = torch.zeros(x.shape[0], self.FrameNum, self.PatchSize*2).cuda()
        for i in range(self.FrameNum):
            Start = i * self.Stride
            End = Start + self.PatchSize
            Input_Feature[:, i] = torch.cat((x[:, Start: End, 0], x[:, Start: End, 1]), dim=-1)
        return self.Embeddinbg(Input_Feature)
        # return self.Embeddinbg(x.permute(0, 2, 1)).permute(0, 2, 1)


class Transformer_layer(nn.Module):
    def __init__(self,
                 d_model=64,
                 d_fix_qk=64,
                 d_fix_v=16,
                 d_mid=256,
                 n_head_qk=4,
                 n_head_v=4,
                 dropout=0.,
                 bias=False,
                 talking=True,
                 real_former=True,
                 activate='relu',
                 ffn_type='GLU'
                 ):
        super(Transformer_layer, self).__init__()
        self.attn_res = real_former
        self.mhsa = MHSA_block(d_model=d_model,
                               d_fix_qk=d_fix_qk,
                               d_fix_v=d_fix_v,
                               n_head_qk=n_head_qk,
                               n_head_v=n_head_v,
                               dropout=dropout,
                               bias=bias,
                               talking=talking,
                               attn_res=real_former)
        if ffn_type == 'MLP':
            self.ffn = MLP_block(d_model=d_model,
                                 dim_feedforward=d_mid,
                                 dropout=dropout,
                                 activate=activate)
        elif ffn_type == "GLU":
            self.ffn = GLU_block(d_model=d_model,
                                 dim_feedforward=d_mid,
                                 dropout=dropout,
                                 activate=activate)
        else:
            self.ffn = DB_GLU_block(d_model=d_model,
                                    dim_feedforward=d_mid,
                                    dropout=dropout,
                                    activate=activate,
                                    merge_method=ffn_type[7:])
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            xavier_uniform_(m.weight)
        if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_=None):
        if self.attn_res:
            x_sa, attn_ = self.mhsa(x, attn_)
        else:
            x_sa = self.mhsa(x)
        x = self.norm_1(x + x_sa)
        x = self.norm_2(x + self.ffn(x))
        return x, attn_


class FEA_T(nn.Module):
    def __init__(self,
                 patch_size=16,  # Frame length L (default value: 32)
                 d_fix_qk=16,  # The embedding dimension of queries and keys of each head d_p (default value: 16)
                 d_fix_v=16,  # The embedding dimension of values of each head d_p (default value: 16)
                 seq_length=128,  # The length of the input signal sequence L (default value: 16)
                 hidden_features=64 * 4,  # The dimension of inner-layer of FFN (default value: 4L for DB-GLU)
                 n_head_qk=4,  # The head number of queries and keys h (default value: 4)
                 n_head_v=4,  # The head number of queries and keys h (default value: 4)
                 overlap=0.2,  # Sliding step length R (default value: 0.5).
                 dropout=0.,  # The dropout ratio for MHSA and FFN
                 layer_num=8,  # The number of transformer layer M (default value: 4)
                 num_class=11,  # The number of modulation types (default value: 24 for RML 2018.01A)
                 pos_emb=True,  # If implement the position embedding (default value: True)
                 bias=False,  # If implement a learnable bias for all linear projection of the MHSA block
                 talking=False,  # If implement the talking-head attention technique of the MHSA block
                 real_former=False,  # If implement the RealFormer technique (We provide this exploration, but not utilized in FEA-T)
                 ffn_type='DB_GLU_add',  # FFN scheme ( GLU | MLP | DB_GLU_add | DB_GLU_cross_add | DB_GLU_cat ) (default value: DB_GLU_add)
                 activation='gelu',  # The activation function of FFN ( relu | sigmoid | gelu ) (default value: gelu)
                 feature_mode='IQ',  # The inpue state of the input signal ( IQ | AP ) (default value: IQ)
                 ):
        super(FEA_T, self).__init__()
        self.if_pos_emb = pos_emb
        self.feature_mode = feature_mode
        n_patch = int((seq_length - patch_size) / int(patch_size * overlap) + 2)
        in_features = patch_size * 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_size * 2))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patch, in_features))
        self.Embedding = Frame(patch_size, n_patch, bias=False, overlap=overlap)
        self.Enc = nn.ModuleList([])
        for i in range(layer_num):
            self.Enc.append(Transformer_layer(d_model=in_features,
                                              d_fix_qk=d_fix_qk,
                                              d_fix_v=d_fix_v,
                                              d_mid=hidden_features,
                                              n_head_qk=n_head_qk,
                                              n_head_v=n_head_v,
                                              dropout=dropout,
                                              bias=bias,
                                              talking=talking,
                                              real_former=real_former,
                                              activate=activation,
                                              ffn_type=ffn_type,))
        self.classifier = nn.Linear(in_features, num_class)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, state=None):
        x = torch.transpose(x,1,2)
        # preprossess
        B = x.shape[0]
        dlY = self.Embedding(x)  # dlY: (B, 1024, 64)
        # cls_token + pos_Emb
        cls_token = self.cls_token.expand(B, -1, -1)
        dlY = torch.cat((dlY, cls_token), dim=1)
        dlY = dlY + self.pos_embed if self.if_pos_emb else dlY
        # ViT process
        attn_res = None
        for i in range(len(self.Enc)):
            dlY, attn_res = self.Enc[i](dlY, attn_res)
        return self.classifier(dlY[:, -1])

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num

if __name__ == '__main__':
    x = torch.randn(4, 2,128)
    Net = FEA_T()

    y = Net(x)
    print(y.shape)
    print(numParams(Net))