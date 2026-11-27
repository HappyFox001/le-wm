"""Token-level JEPA world models for LeWM: a dense token baseline and G1 (sparse residual, energy-gated).

LeWM represents each frame with one CLS vector. A per-token change gate needs spatial tokens, so these
models use the ViT patch tokens (optionally average-pooled), each passed through a per-token projector
with BatchNorm (the same reason LeWM projects its CLS token: the ViT's final LayerNorm blocks SIGReg).

mode
  dense : s_hat_{t+1} = pred_proj(h)                    (LeWM, but on tokens)
  res   : s_hat_{t+1} = s_t + head(h), head zero-init   (dense residual, no gate)
  g1    : s_hat_{t+1,i} = s_{t,i} + g_i * d_i,  g_i = 1[logit_i > 0]
          Trained by energy-minimising inference of the gate (see train.py: g1_losses).
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from jepa import JEPA, detach_clone
from module import FeedForward, modulate


class MaskedAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        x = self.norm(x)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class MaskedConditionalBlock(nn.Module):
    """LeWM's AdaLN-zero block, with a block-causal (frame-level) attention mask."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MaskedAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, mask):
        sm, cm, gm, sf, cf, gf = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gm * self.attn(modulate(self.norm1(x), sm, cm), mask)
        x = x + gf * self.mlp(modulate(self.norm2(x), sf, cf))
        return x


class TokenPredictor(nn.Module):
    """Autoregressive predictor over (frames x tokens). Tokens of frame t attend to all tokens of frames <= t.
    The action embedding of frame t conditions every token of frame t through AdaLN."""

    def __init__(self, *, num_frames, num_tokens, depth, heads, mlp_dim, input_dim, hidden_dim,
                 dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.frame_pos = nn.Parameter(torch.randn(1, num_frames, 1, input_dim) * 0.02)
        self.token_pos = nn.Parameter(torch.randn(1, 1, num_tokens, input_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [MaskedConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, c):
        """x: (B, T, N, D), c: (B, T, D) -> (B, T, N, hidden)"""
        B, T, N, _ = x.shape
        x = self.dropout(x + self.frame_pos[:, :T] + self.token_pos)
        x = self.input_proj(rearrange(x, "b t n d -> b (t n) d"))
        c = self.cond_proj(c).unsqueeze(2).expand(-1, -1, N, -1).reshape(B, T * N, -1)
        f = torch.arange(T, device=x.device).repeat_interleave(N)
        mask = f[:, None] >= f[None, :]  # True = may attend
        for blk in self.layers:
            x = blk(x, c, mask)
        return rearrange(self.norm(x), "b (t n) d -> b t n d", t=T)


class TokenJEPA(JEPA):
    def __init__(self, encoder, predictor, action_encoder, projector, pred_proj=None,
                 mode="g1", pool=2, hidden_dim=192, embed_dim=192):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        assert mode in ("dense", "res", "g1")
        self.mode = mode
        self.pool = pool
        if mode in ("res", "g1"):
            self.head_d = nn.Linear(hidden_dim, embed_dim)
            nn.init.zeros_(self.head_d.weight)
            nn.init.zeros_(self.head_d.bias)
        if mode == "g1":
            self.head_pi = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.head_pi.weight)
            nn.init.constant_(self.head_pi.bias, -3.0)

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        tok = self.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 1:]  # drop CLS
        n = int(math.isqrt(tok.size(1)))
        if self.pool > 1:
            tok = rearrange(tok, "bt (h w) d -> bt d h w", h=n)
            tok = rearrange(F.avg_pool2d(tok, self.pool), "bt d h w -> bt (h w) d")
        bt, N, D = tok.shape
        emb = self.projector(tok.reshape(bt * N, D)).reshape(bt, N, -1)
        info["emb"] = rearrange(emb, "(b t) n d -> b t n d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict_aux(self, emb, act_emb):
        """emb: (B, T, N, D). Returns dict with s_next (B, T, N, D) and, for g1, d / logit."""
        h = self.predictor(emb, act_emb)
        if self.mode == "dense":
            B, T, N, H = h.shape
            return {"s_next": self.pred_proj(h.reshape(-1, H)).reshape(B, T, N, -1)}
        d = self.head_d(h)
        if self.mode == "res":
            return {"s_next": emb + d, "d": d}
        logit = self.head_pi(h).squeeze(-1).float()
        gate = (logit > 0).to(d.dtype).unsqueeze(-1)  # hard gate: closed tokens are copied exactly
        return {"s_next": emb + gate * d, "d": d, "logit": logit}

    def predict(self, emb, act_emb):
        return self.predict_aux(emb, act_emb)["s_next"]

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Same as JEPA.rollout, generalised to token embeddings (B, S, T, N, D)."""
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, *_init["emb"].shape[1:])
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps + 1):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            if t < n_steps:
                act = torch.cat([act, act_future[:, t: t + 1, :]], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        """Last-step squared distance to the goal, summed over tokens and dims. -> (B, S)"""
        pred = info_dict["predicted_emb"][:, :, -1]  # (B, S, N, D)
        goal = info_dict["goal_emb"][:, -1].unsqueeze(1)  # (B, 1, N, D)
        return (pred - goal.detach()).pow(2).sum(dim=(-1, -2))


def g1_losses(model, ctx_emb, tgt_emb, aux, cfg):
    """Energy-inferred gate (two-part MDL code), in LeWM's units (mean over feature dims).

    c_i = s_{t+1,i} - s_{t,i}
    E_i(m) = m * |c_i - d_i|^2 + (1 - m) * |c_i|^2 + m * beta
    m*_i   = 1[ |c_i|^2 - |c_i - d_i|^2 > beta ]     (no gradient through m*)
    """
    c = (tgt_emb - ctx_emb).float()
    d = aux["d"].float()
    D = c.size(-1)
    err = (c - d).pow(2).mean(-1)
    e_off = c.pow(2).mean(-1)
    with torch.no_grad():
        # scale: median distance between same-position tokens of different samples
        s = ctx_emb.float()
        rho = (s - s.roll(1, dims=0)).norm(dim=-1).median().clamp(min=1e-6)
        beta = cfg.beta * rho ** 2 / D
        mstar = ((e_off - err) > beta).float()
    e_off_term = e_off if cfg.enc_route == "all" else torch.zeros_like(e_off)
    L_energy = (mstar * err + (1 - mstar) * e_off_term).mean()
    L_dwarm = (d - c.detach()).pow(2).mean()
    L_focus = F.binary_cross_entropy_with_logits(
        aux["logit"], mstar, pos_weight=torch.tensor(cfg.pos_weight, device=c.device))
    loss = L_energy + cfg.lam_dwarm * L_dwarm + (cfg.focus_weight / D) * L_focus
    stats = {
        "g1_energy": L_energy.detach(), "g1_dwarm": L_dwarm.detach(), "g1_focus": L_focus.detach(),
        "g1_mstar_frac": mstar.mean(), "g1_active_frac": (aux["logit"] > 0).float().mean(),
        "g1_rho": rho, "g1_change_rms": e_off.mean().sqrt().detach(),
    }
    return loss, stats
