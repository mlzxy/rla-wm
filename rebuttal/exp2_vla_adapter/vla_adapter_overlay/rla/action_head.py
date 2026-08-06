"""
rla/action_head.py

The policy head with RLA output-query tokens, as three subclasses of the upstream modules.

    x_action : (B, 8, 7*D) -> layer_norm1 -> fc1 -> relu -> (B, 8, D)
    x_rla    : (B, 8*16, D)  zeros (+ training noise); starts at D, so no fc1
    x        = cat([x_action, x_rla], dim=1)                    # (B, 136, D)
       ... 24 x MLPResNetBlock_Pro over the combined sequence ...
    action   = fc2(layer_norm2(x[:, :8]))                       # (B, 8, 7)
    z_hat    = fc2_rla(layer_norm2_rla(x[:, 8:]))               # (B, 128, 64) -> (B, 8, 1024)

Three things about the upstream code make this a small change rather than a rewrite:

  * `T = x.size(1)` is read dynamically inside every block, so nothing in the attention math assumes
    a fixed number of query tokens.
  * `MLPResNetBlock_Pro` gates the vision stream with a single `tanh(gating_factor)` scalar that has
    exactly one consumer, `matmul(q_1, k_task.T) * ratio_g` producing `(B, H, T, K_t)`
    (`action_heads.py:344, :391`). Widening the parameter to `(T, 1)` broadcasts as `(1, 1, T, 1)`
    and gives per-position gating -- action tokens and RLA tokens learn independent "how much vision
    to let in" schedules -- with **no copied forward code**.
  * `MLPResNet.forward` is the only place the token sequence is built and split, so keeping `z_hat`
    on the module instead of in the return value means `L1RegressionActionHead.predict_action` needs
    no override at all. That matters: `prismatic/extern/hf/modeling_prismatic.py:871` calls
    `predict_action(...)` and immediately `.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)`, and that file
    also exists as a copy under `pretrained_models/configs/` which `check_model_logic_mismatch`
    installs into every checkpoint dir. Changing the return type would mean editing both.

One combined self-attention pass, deliberately. Running action-as-Q and rla-as-Q as two passes would
silently drop the cross-group attention unless K/V were re-merged in both.

With `RLA_STEPS=0` every module below is byte-identical to upstream -- same block class, same
state-dict keys, same shapes -- so released and existing `LIBERO-*-Pro` checkpoints still load
strict. `rla/tests/` gate 3 asserts it rather than trusting it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from prismatic.models.action_heads import (
    L1RegressionActionHead,
    MLPResNet,
    MLPResNetBlock_Pro,
    apply_rope,
    learnable_random_perturbations,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from rla.config import CFG


class RlaMLPResNetBlockPro(MLPResNetBlock_Pro):
    """`MLPResNetBlock_Pro` with a per-query-position vision gate and a one-way self-attention mask.

    **Gate.** `ratio_g = torch.tanh(self.gating_factor)` scales the task/vision attention scores of
    shape `(B, H, T, K_t)`; a `(T, 1)` parameter broadcasts against that correctly, so positions
    `< num_action_tokens` and positions `>=` it get separate, independently-learned gates.
    Zero-init, exactly like the scalar it replaces. `__init__` alone handles this.

    **Mask.** Attention between the two groups is made one-way: action queries may attend to RLA
    keys, RLA queries may **not** attend to action keys. So the RLA prediction is a function of
    (vision, adapter, proprio, other RLA tokens) only, and is completely independent of the action
    tokens -- which is what keeps the RLA branch bit-identical from stage 1 to stage 2 even though
    stage 2 newly supervises `fc2` and reshapes what the action tokens carry. Without it, turning on
    the action loss would silently move the auxiliary task's own inputs.

    **Bridge gate on the surviving direction.** `gating_factor_rla2act` is a per-block scalar that
    scales the scores of **action queries attending to RLA keys** -- the one cross-group direction
    the mask leaves open. Zero-init, so it starts exactly where upstream's vision gate starts, and
    the network can learn how much the predicted latent action should inform the action chunk. Note
    the semantics it inherits from `gating_factor`: `tanh(g) = 0` flattens those logits to a
    constant, so the RLA keys become *uninformative* rather than unattended -- the same thing
    upstream's zero-init vision gate does at step 0.

    Both need the score matrix, which the parent never exposes, so `forward` below is a verbatim
    copy of `MLPResNetBlock_Pro.forward` (`action_heads.py:337-410`) with two added statements,
    marked. The masks are `persistent=False` buffers, so only the two parameters enter the state
    dict.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        seq_len: int = NUM_ACTIONS_CHUNK,
        num_action_tokens: int = 0,
        mask_rla_to_action: bool = True,
    ):
        super().__init__(dim, num_heads=num_heads)
        self.gating_factor = nn.Parameter(torch.zeros(seq_len, 1))
        self.gating_factor_rla2act = nn.Parameter(torch.zeros(1))

        cross = torch.zeros(seq_len, seq_len)          # action queries x RLA keys
        block = None                                   # RLA queries x action keys, blocked
        if 0 < num_action_tokens < seq_len:
            cross[:num_action_tokens, num_action_tokens:] = 1.0
            if mask_rla_to_action:
                block = torch.zeros(seq_len, seq_len)
                block[num_action_tokens:, :num_action_tokens] = float("-inf")
        self.register_buffer("rla2act_mask", cross, persistent=False)
        self.register_buffer("rla_self_mask", block, persistent=False)

    def forward(self, x, h_a=None, h_t=None, p=None):
        """Verbatim `MLPResNetBlock_Pro.forward`, plus the one marked statement."""
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p), dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)

        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        # >>> the two changes vs MLPResNetBlock_Pro, both on the policy self-attention block:
        #     (1) gate how much the RLA keys inform the action queries,
        #     (2) forbid the reverse direction outright.
        ratio_ra = torch.tanh(self.gating_factor_rla2act)
        cross = self.rla2act_mask[:T, :T].to(attn_scores[0].dtype)
        attn_scores[0] = attn_scores[0] * (1.0 + (ratio_ra - 1.0) * cross)
        if self.rla_self_mask is not None:
            attn_scores[0] = attn_scores[0] + self.rla_self_mask[:T, :T].to(attn_scores[0].dtype)
        # <<<
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens, v_adapter, v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # residual + FFN
        x = self.ffn(output + x)
        return x


class RlaMLPResNet(MLPResNet):
    """`MLPResNet` with `num_rla_tokens` extra output-query tokens and a second output head."""

    def __init__(
        self,
        num_blocks,
        input_dim,
        hidden_dim,
        output_dim,
        use_pro_version=False,
        num_rla_tokens: int = 0,
        rla_steps: int = 0,
        rla_dim: int = 0,
        action_tokens: bool = True,
        mask_rla_to_action: bool = True,
    ):
        # Skip MLPResNet.__init__ and rebuild it: it hard-codes the block class, and constructing a
        # 24-block trunk twice just to throw one away costs ~600 MB and several seconds.
        nn.Module.__init__(self)

        if num_rla_tokens and not use_pro_version:
            raise ValueError(
                "RLA tokens require use_pro_version=True. `x` enters the policy as zeros and the "
                "non-Pro MLPResNetBlock has neither RoPE nor a positional embedding, so all of its "
                "query positions are mathematically identical -- every RLA token would predict the "
                "same z. (The same pathology makes the non-Pro head emit one action eight times.)"
            )

        self.num_rla_tokens = num_rla_tokens
        self.rla_steps = rla_steps
        self.rla_dim = rla_dim
        # False = stage-1-RLA-only: the action tokens are built (so every parameter stays in the
        # graph and DDP sees no unused ones) but kept OUT of the policy sequence. Only legal with
        # lambda_act=0, which rla/config.py enforces.
        self.action_tokens = bool(action_tokens) or not num_rla_tokens
        self._z_hat = None

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        num_action_tokens = NUM_ACTIONS_CHUNK if self.action_tokens else 0
        seq_len = num_action_tokens + num_rla_tokens
        self.mask_rla_to_action = bool(mask_rla_to_action)
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_rla_tokens:
                self.mlp_resnet_blocks.append(
                    RlaMLPResNetBlockPro(
                        dim=hidden_dim,
                        seq_len=seq_len,
                        num_action_tokens=num_action_tokens,
                        mask_rla_to_action=self.mask_rla_to_action,
                    )
                )
            elif use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                from prismatic.models.action_heads import MLPResNetBlock

                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Created only when RLA is on, so the RLA_STEPS=0 state dict stays upstream's.
        if num_rla_tokens:
            self.layer_norm2_rla = nn.LayerNorm(hidden_dim)
            self.fc2_rla = nn.Linear(hidden_dim, rla_dim)

    def pop_z_hat(self):
        """Take the last forward's `(B, rla_steps, Q*D)` prediction and drop the reference.

        `forward` returns only the action tensor -- see the module docstring for why -- and leaves
        `z_hat` here. Popped rather than read so the autograd graph is not held into the next step.
        """
        z_hat, self._z_hat = self._z_hat, None
        return z_hat

    def forward(self, x, h_a=None, h_t=None, p=None):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)                                   # (B, NUM_ACTIONS_CHUNK, hidden)

        if not self.num_rla_tokens:
            for i, block in enumerate(self.mlp_resnet_blocks):
                x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p)
            return self.fc2(self.layer_norm2(x))

        x_pre_action = x                                   # (B, NUM_ACTIONS_CHUNK, hidden)
        num_action_tokens = x_pre_action.shape[1]

        # Zeros plus the same per-step Gaussian input noise the action tokens get during training
        # (`predict_action` applies it before fc1; these tokens start at hidden_dim, so it is applied
        # here instead). Symmetric treatment of the two groups.
        x_rla = torch.zeros(
            x.shape[0], self.num_rla_tokens, x.shape[-1], device=x.device, dtype=x.dtype
        )
        if self.training:
            x_rla = x_rla + learnable_random_perturbations(
                self.num_rla_tokens, x.shape[-1], device=x.device, dtype=x.dtype
            )

        x = torch.cat([x_pre_action, x_rla], dim=1) if self.action_tokens else x_rla

        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p)

        if self.action_tokens:
            x_action, x_rla = x[:, :num_action_tokens], x[:, num_action_tokens:]
        else:
            # The trunk never saw the action tokens, so there is no trunk output to read them from.
            # Project the pre-trunk tokens instead: the value is meaningless (lambda_act is 0), but
            # it keeps fc2/layer_norm2 in the autograd graph so they get a zero grad rather than
            # None, which is what stops DDP reporting an unused parameter.
            x_action, x_rla = x_pre_action, x

        action = self.fc2(self.layer_norm2(x_action))      # (B, NUM_ACTIONS_CHUNK, action_dim)

        z_hat = self.fc2_rla(self.layer_norm2_rla(x_rla))              # (B, steps*Q, D)
        self._z_hat = z_hat.reshape(z_hat.shape[0], self.rla_steps, -1)  # (B, steps, Q*D)
        return action


class RlaL1RegressionActionHead(L1RegressionActionHead):
    """`L1RegressionActionHead` whose trunk carries the RLA tokens.

    Constructor signature is identical to upstream's, on purpose: `finetune.init_module` and
    `experiments/robot/openvla_utils.get_action_head` both build the head from a fixed kwarg dict,
    so the RLA geometry comes from the environment (`rla/config.py`) instead. That is also what
    makes the strict load in `get_action_head` a useful tripwire -- a train/eval mismatch in
    `RLA_STEPS` fails at load instead of silently randomising the head.
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
    ):
        nn.Module.__init__(self)
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = RlaMLPResNet(
            num_blocks=24,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_pro_version=use_pro_version,
            num_rla_tokens=CFG.num_tokens,
            rla_steps=CFG.steps,
            rla_dim=CFG.dim,
            action_tokens=CFG.action_tokens,
            mask_rla_to_action=CFG.mask_rla_to_action,
        )

    # `predict_action` is inherited unchanged -- see the module docstring.

    def pop_z_hat(self):
        return self.model.pop_z_hat()
