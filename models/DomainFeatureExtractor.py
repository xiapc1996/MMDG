import torch

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv):
        x_q = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)

        q = self.to_q(x_q)
        kv = self.to_kv(x_kv).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # return self.to_out(out)
        return out

class DomainFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DomainFeatureExtractor, self).__init__()
        self.dim = input_dim
        self.cam_vib = Cross_Attention(dim=self.dim, heads=8, dim_head=32)
        self.cam_cur = Cross_Attention(dim=self.dim, heads=8, dim_head=32)
        self.cam_aud = Cross_Attention(dim=self.dim, heads=8, dim_head=32)
        self.fuse_trans = nn.Linear(256*3, 256, bias = False)
        self.relu = nn.ReLU()
        self.layer = nn.Sequential(
            nn.Linear(input_dim*3, output_dim),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )        

    def forward(self, feat_vib, feat_cur, feat_aud):
        # feat = torch.cat((feat_vib, feat_cur, feat_aud), dim=-1)

        feat_vib, feat_cur, feat_aud = feat_vib.unsqueeze(-2), \
                                        feat_cur.unsqueeze(-2), \
                                        feat_aud.unsqueeze(-2)
        fused_vib_cur = self.cam_vib(feat_vib, feat_cur)
        fused_cur_aud = self.cam_cur(feat_cur, feat_aud)
        fused_aud_vib = self.cam_aud(feat_aud, feat_vib)

        fused_out = self.fuse_trans(torch.cat((fused_vib_cur.squeeze(-2), fused_cur_aud.squeeze(-2), fused_aud_vib.squeeze(-2)), -1))
        # fused_out = self.fuse_trans(torch.mean(torch.stack((fused_vib_cur, fused_cur_aud, fused_aud_vib), dim=-2), dim=-2).squeeze(-2))
        # fused_out = self.fuse_trans(fused_vib_cur.squeeze(-2) + fused_cur_aud.squeeze(-2) + fused_aud_vib.squeeze(-2))

        return fused_out
        # return feat
