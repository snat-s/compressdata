"""
RWKV-v7 implementation for neural compression.
Based on https://github.com/BlinkDL/RWKV-LM (RWKV-v7)

Pure PyTorch implementation - no custom CUDA kernels required.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class RWKVv7Config:
    block_size: int = 1024
    vocab_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RWKV_v7_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        n_embd = config.n_embd
        n_head = config.n_head
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(config.n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(n_head)
            for h in range(n_head):
                decay_speed[h] = -6 + 5 * (h / max(n_head - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, n_head, 1).expand(1, 1, n_head, self.head_size).clone())
            
            self.a0 = nn.Parameter(torch.zeros(1, 1, n_embd))
            self.v0 = nn.Parameter(torch.zeros(1, 1, n_embd) + 1.0)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.key = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.value = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.gate = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.output = nn.Linear(n_embd, n_embd, bias=config.bias)
        
        self.w1 = nn.Linear(n_embd, n_embd, bias=False)
        self.w2 = nn.Parameter(torch.ones(1, 1, n_embd) * 0.05)
        self.a1 = nn.Linear(n_embd, n_embd, bias=False)
        self.a2 = nn.Parameter(torch.zeros(n_embd))
        self.v1 = nn.Linear(n_embd, n_embd, bias=False)
        self.v2 = nn.Parameter(torch.zeros(n_embd))
        
        self.layer_id = layer_id
        self.g1 = nn.Linear(n_embd, n_embd, bias=False)
        self.g2 = nn.Parameter(torch.zeros(n_embd) + 1.0)
        
        self.k_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.85)
        self.k_a = nn.Parameter(torch.ones(1, 1, n_embd))
        self.r_k = nn.Parameter(torch.zeros(n_head, self.head_size))
        
        self.ln_x = nn.GroupNorm(n_head, n_embd, eps=64e-5)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, v_first=None):
        B, T, C = x.size()
        H, N = self.n_head, self.head_size

        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = self.w0 + (torch.tanh(self.w1(xw)) * self.w2).view(B, T, H, N)
        w = -F.softplus(-w) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + self.v1(xv) * self.v2)
        
        a = torch.sigmoid(self.a0 + self.a1(xa) * self.a2)
        g = torch.sigmoid(self.g1(xg)) * self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        r = r.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        kk = kk.view(B, T, H, N)
        w = w.view(B, T, H, N)
        a = a.view(B, T, H, N)

        w = torch.exp(w.clamp(-10, 0))

        state = torch.zeros(B, H, N, N, device=x.device, dtype=x.dtype)
        y = torch.zeros(B, T, H, N, device=x.device, dtype=x.dtype)

        for t in range(T):
            rt = r[:, t]
            kt = k[:, t]
            vt = v[:, t]
            kkt = kk[:, t]
            wt = w[:, t]
            at = a[:, t]

            vk = vt.unsqueeze(-1) * kt.unsqueeze(-2)
            ab = (-kkt).unsqueeze(-1) * (kkt * at).unsqueeze(-2)
            state = state * wt.unsqueeze(-1) + state @ ab + vk

            y[:, t] = (state @ rt.unsqueeze(-1)).squeeze(-1)

        y = y.view(B * T, C)
        y = self.ln_x(y).view(B, T, C)
        
        y = y + ((r * k * self.r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True) * v).view(B, T, C)
        y = y * g

        y = self.dropout(self.output(y))
        return y, v_first


class RWKV_v7_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        n_embd = config.n_embd

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, n_embd * 4, bias=config.bias)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.x_k
        k = self.key(xk)
        k = F.relu(k) ** 2
        return self.dropout(self.value(k))


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.tmix = RWKV_v7_TimeMix(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cmix = RWKV_v7_ChannelMix(config, layer_id)

    def forward(self, x, v_first=None):
        tmix_out, v_first = self.tmix(self.ln_1(x), v_first)
        x = x + tmix_out
        x = x + self.cmix(self.ln_2(x))
        return x, v_first


class RWKVv7LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("RWKV-v7 parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        v_first = None
        for block in self.transformer.h:
            x, v_first = block(x, v_first)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def full_loss(self, inputs, with_grad=True):
        logits, _ = self.forward(inputs[:, :-1])
        logits = logits.transpose(1, 2)
        
        loss = F.cross_entropy(logits[:, :, -1], inputs[:, -1], reduction='mean')
        if with_grad:
            loss.backward()
        return loss, logits
