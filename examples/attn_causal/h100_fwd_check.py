import torch 
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-312')
import h100_fwd as mod

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def pytorch_test(q, k, v, tau, g4, causal=True):
    # output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    src_len = k.size(2)
    tgt_len = q.size(2)
    softmax_scale = q.shape[-1] ** -0.5
    # tau = 1 + self.gate_tau
    attn_weights = torch.matmul(q, k.transpose(-1, -2))
    attn_weights = attn_weights * softmax_scale
    g1 = attn_weights
    # g4 = torch.sum(self.gate_b_q * self.gate_b_k, dim=-1, keepdim=True) - 1.0
    offset = src_len - tgt_len
    if causal:
        attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )

        attn_gate = F.sigmoid( (g1 + g4) / tau + attn_mask)
    else:
        attn_gate = F.sigmoid( (g1 + g4) / tau)
    attn_gate = attn_gate / (attn_gate.detach().max(dim=-1, keepdim=True)[0] + 1e-5)
    if causal:
        attn_weights += attn_mask  
    attn_weights = (F.softmax(attn_weights, dim=-1).type_as(attn_weights) * attn_gate).to(q)

    attn = torch.matmul(attn_weights, v)
    
    return attn
    # return output

def h100_fwd_kernel_test(Q, K, V, tau, g4):
    o = torch.zeros_like(Q)
    mod.attention_forward_causal(Q, K, V, tau, g4, o)
    return o

def check_correctness(b, h, n, d):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}")
    
    Q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    K = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    V = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    gate_b_q = nn.Parameter(torch.zeros(1, h, 1, d, dtype=torch.float32, device='cuda:0').normal_(mean=0,std=0.1))
    gate_b_k = nn.Parameter(torch.zeros(1, h, 1, d, dtype=torch.float32, device='cuda:0').normal_(mean=0,std=0.1))
    gate_tau = nn.Parameter(torch.zeros(1, h, 1, 1, dtype=torch.float32, device='cuda:0'))
    tau = 1 + gate_tau
    g4 = torch.sum(gate_b_q * gate_b_k, dim=-1, keepdim=True) - 1.0
    tau = tau.to(torch.bfloat16)
    g4 = g4.to(torch.bfloat16)
    result_pytorch = pytorch_test(Q, K, V, tau, g4)
    tk_result = h100_fwd_kernel_test(Q, K, V, tau, g4)
    
    diff = result_pytorch - tk_result
    avg_diff_mag = torch.mean(torch.abs(diff)).item()
    avg_diff_per = 100 * avg_diff_mag / torch.mean(torch.abs(result_pytorch)).item()
    
    print(f"Attention output - avg magnitude of diff: {avg_diff_mag:.6f}")
    print("-" * 40)

print("Correctness Tests: ")
configurations = [
    (2,  8, 256,   64),
    (4,  8, 512,   64),
    (8,  8, 1024,  64),
    (16, 8, 2048,  64),
    (16, 8, 4096,  64),
    (16, 8, 8192,  64),
    (16, 8, 16384, 64),
    (2,  8, 256,   128),
    (4,  8, 512,   128),
    (8,  8, 1024,  128),
    (16, 8, 2048,  128),
    (16, 8, 4096,  128),
    (16, 8, 8192,  128),
    (16, 8, 16384, 128)
]
for b, h, n, d in configurations:
    check_correctness(b, h, n, d)
