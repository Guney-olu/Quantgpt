import os
import math
import time
from dataclasses import dataclass, asdict
from itertools import chain
import copy
import random
import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import save_model, save_file
import pyarrow.parquet as pq
import tiktoken

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# WANDB (for logging)
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"DDP setup: Rank {rank}/{world_size} on device {device}")
    return rank, world_size, device

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_theta_pos_frequencies(head_dim, max_seq_len, theta=10000.0, device="cpu"):
    assert head_dim % 2 == 0
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs, dtype=torch.float32), freqs).to(torch.complex64)

def apply_rotary_embeddings(x, freqs_complex):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_kv_head, self.n_embd = config.n_head, config.n_kv_head, config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, freqs_complex):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
        q = apply_rotary_embeddings(q, freqs_complex)
        k = apply_rotary_embeddings(k, freqs_complex)
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k, v = k.repeat_interleave(n_rep, dim=2), v.repeat_interleave(n_rep, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, C))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1, self.attn = RMSNorm(config.n_embd), CausalSelfAttention(config)
        self.ln_2, self.mlp = RMSNorm(config.n_embd), FeedForward(config)

    def forward(self, x, freqs_complex):
        x = x + self.attn(self.ln_1(x), freqs_complex)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class MobileLLMConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    n_kv_head: int = 4

class MobileLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        freqs = precompute_theta_pos_frequencies(self.config.n_embd // self.config.n_head, self.config.block_size)
        self.register_buffer("freqs_complex", freqs, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        freqs = self.freqs_complex[:T]
        for blk in self.transformer.h:
            x = blk(x, freqs)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# DDP-AWARE Parquet Data Loader

class ParquetDataLoaderLite:
    def __init__(self, B, T, data_root, split, rank=0, world_size=1, shuffle=True):
        self.B, self.T, self.shuffle = B, T, shuffle
        self.rank = rank
        self.world_size = world_size
        assert split in {"train", "val"}
        shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s and s.endswith(".parquet")])
        assert len(shards) > 0, f"No parquet shards found for split {split}"
        self.all_shards = shards
        if self.rank == 0:
            print(f"Found {len(shards)} parquet shards for split '{split}'")
        self.reset()

    def reset(self):
        self.shards = self.all_shards[:]
        if self.shuffle:
            random.shuffle(self.shards)
        dist.broadcast_object_list(self.shards, src=0)
        self.current_shard_idx = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard_idx])
        self.current_position = 0
        if self.rank == 0:
            print(f"Loader reset. Rank 0 starting with shard: {self.shards[self.current_shard_idx]}")

    def _load_tokens(self, filename):
        table = pq.read_table(filename, columns=["tokens"])
        return torch.tensor(list(chain.from_iterable(table["tokens"].to_pylist())), dtype=torch.long)

    def _advance_shard(self):
        self.current_shard_idx += 1
        if self.current_shard_idx >= len(self.shards):
            if self.rank == 0:
                print("Completed an epoch over all data shards. Re-shuffling...")
            self.reset()
        else:
            self.tokens = self._load_tokens(self.shards[self.current_shard_idx])
            self.current_position = 0
            if self.rank == 0:
                print(f"Rank 0 switching to shard: {self.shards[self.current_shard_idx]}")

    def next_batch(self):
        B, T = self.B, self.T
        buf_len = B * T + 1
        while self.current_position + buf_len > len(self.tokens):
            self._advance_shard()
        buf = self.tokens[self.current_position : self.current_position + buf_len]
        self.current_position += B * T
        x_full, y_full = buf[:-1].view(B, T), buf[1:].view(B, T)
        x = x_full[self.rank::self.world_size]
        y = y_full[self.rank::self.world_size]
        return x, y

# Training Loop

# DDP: Initialize process group
rank, world_size, device = ddp_setup()

torch.manual_seed(1337 + rank)

config = MobileLLMConfig()
model = MobileLLM(config).to(device)
model = DDP(model, device_ids=[device])
raw_model = model.module
if rank == 0:
    print(f"Model created with {sum(p.numel() for p in raw_model.parameters()):,} parameters.")

# Training settings
max_steps = 50000
micro_batch_size = 64
grad_accum_steps = 2
effective_batch_size = micro_batch_size * grad_accum_steps
if rank == 0:
    print(f"Effective batch size: {effective_batch_size} (micro: {micro_batch_size} x {grad_accum_steps} accum)")
    print(f"Per-GPU batch size: {micro_batch_size // world_size}")

max_lr, min_lr = 6e-4, 6e-5
warmup_steps = 100
eval_interval = 500
log_interval = 100
checkpoint_interval = 5000

amp_enabled = True
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))

if rank == 0:
    wandb.init(
        project="mobile-llm-fineweb-long",
        name=f"mobile-llm-{int(time.time())}",
        config={
            **asdict(config),
            "max_steps": max_steps,
            "learning_rate": max_lr,
            "min_lr": min_lr,
            "effective_batch_size": effective_batch_size,
            "grad_accum_steps": grad_accum_steps,
        }
    )

DATA_CACHE_DIR = "../train_model/edu_fineweb10B" #Folder which has the data
train_loader = ParquetDataLoaderLite(B=micro_batch_size, T=config.block_size, data_root=DATA_CACHE_DIR, split="train", rank=rank, world_size=world_size)
val_loader = ParquetDataLoaderLite(B=micro_batch_size, T=config.block_size, data_root=DATA_CACHE_DIR, split="val", rank=rank, world_size=world_size, shuffle=False)

def get_lr(it):
    if it < warmup_steps: return max_lr * (it + 1) / warmup_steps
    if it > max_steps: return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

if rank == 0:
    print("Starting training...")
    os.makedirs("checkpoints", exist_ok=True)

for step in range(max_steps):
    t0 = time.time()
    is_eval_step = (step % eval_interval == 0) or (step == max_steps - 1)
    is_checkpoint_step = (step > 0 and step % checkpoint_interval == 0) or (step == max_steps - 1)

    if is_eval_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 20
            for _ in range(val_batches):
                x, y = val_loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    _, loss = model(x, y)
                val_loss += loss.item()
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            avg_val_loss = val_loss_tensor.item() / val_batches
            if rank == 0:
                print(f"Validation loss: {avg_val_loss:.4f}")
                wandb.log({"validation/loss": avg_val_loss}, step=step)
        model.train()

    if rank == 0 and is_checkpoint_step:
        checkpoint_path = f"checkpoints/mobile_llm_step_{step}.safetensors"
        save_model(raw_model, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    total_train_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    for i in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_last_accum = (i == grad_accum_steps - 1)
        with model.no_sync() if not is_last_accum else contextlib.nullcontext():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
            total_train_loss += loss.item()
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for pg in optimizer.param_groups: pg["lr"] = lr
    
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    torch.cuda.synchronize()

    if rank == 0 and (step % log_interval == 0 or step == max_steps - 1):
        dt = (time.time() - t0) * 1000
        print(f"Step {step:5d} | Train Loss: {total_train_loss:.4f} | LR: {lr:.2e} | Time: {dt:.2f}ms")
        wandb.log({
            "train/loss": total_train_loss,
            "train/lr": lr,
            "timing/step_time_ms": dt,
        }, step=step)

dist.destroy_process_group()

if rank == 0:
    wandb.finish()
    final_model_path = "../train_model/mobile_llm_final.safetensors"
    last_checkpoint_path = f"checkpoints/mobile_llm_step_{max_steps-1}.safetensors"
    os.rename(last_checkpoint_path, final_model_path)
    print(f"Final model saved to {final_model_path}")

    enc = tiktoken.get_encoding("gpt2")
    raw_model.eval()

    start_text = "The internet is"
    start_ids = torch.tensor(enc.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            generated_ids = raw_model.generate(start_ids, max_new_tokens=100, temperature=0.8, top_k=20)
        generated_text = enc.decode(generated_ids[0].tolist())
        print(f"Prompt: '{start_text}'")
        print(f"Generated text: {generated_text}")
