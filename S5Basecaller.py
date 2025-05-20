import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat


# Core S5 components implementation
class S5SSM(nn.Module):
    """SSM component of the S5 block."""

    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, dt_init='random', dt_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize parameters
        # S4-style discretization parameters
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale

        if dt_init == 'random':
            log_dt = torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        elif dt_init == 'uniform':
            log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), self.d_model)
        else:
            raise NotImplementedError(f"dt_init '{dt_init}' not implemented")

        # Initialize the state space parameters A, B, C
        # A is a diagonal matrix with negative values for stability
        self.log_dt = nn.Parameter(log_dt)

        # Initialize diagonal state matrix A
        A_real = torch.rand(self.d_model, self.d_state) * 2 - 1  # Uniform in [-1, 1]
        A_real = -torch.exp(A_real)  # Make the real parts negative for stability
        self.A_real = nn.Parameter(A_real)

        # Initialize input projection matrix B
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state))

        # Initialize output projection matrix C
        self.C = nn.Parameter(torch.randn(self.d_model, self.d_state))

        # Initial state (optional, set to zeros for now)
        self.x0 = nn.Parameter(torch.zeros(self.d_model, self.d_state))

    def forward(self, u):
        """
        u: Input tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = u.shape
        dt = torch.exp(self.log_dt) * self.dt_scale  # (d_model,)

        # Discretize the continuous system using bilinear transform
        # For diagonal state matrix case: A_tilde = exp(A*dt)
        A_tilde = torch.exp(self.A_real * dt.unsqueeze(-1))  # (d_model, d_state)

        # B_tilde = (exp(A*dt) - I) / A * B (use the approximation for numerical stability)
        B_tilde = (1 - A_tilde) * self.B / self.A_real  # (d_model, d_state)

        # Initialize the hidden state
        x = repeat(self.x0, 'd_model d_state -> b d_model d_state', b=batch)  # (batch, d_model, d_state)

        # Compute the output
        outputs = []
        for t in range(seq_len):
            # Update the state: x_t = A_tilde * x_{t-1} + B_tilde * u_t
            x = A_tilde.unsqueeze(0) * x + B_tilde.unsqueeze(0) * u[:, t, :].unsqueeze(-1)  # (batch, d_model, d_state)

            # Compute the output: y_t = C * x_t
            y = torch.sum(self.C.unsqueeze(0) * x, dim=-1)  # (batch, d_model)
            outputs.append(y)

        # Stack the outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y


class S5Block(nn.Module):
    """S5 Block consisting of SSM, gating, and normalization."""

    def __init__(self, d_model=128, d_state=64, dropout=0.1, gating=True):
        super().__init__()
        self.d_model = d_model
        self.gating = gating

        # SSM layer
        self.ssm = S5SSM(d_model=d_model, d_state=d_state)

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gating mechanism
        if gating:
            self.gate = nn.Linear(d_model, d_model)

        # MLP for residual
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # First normalization
        x_norm = self.norm1(x)

        # Apply SSM
        ssm_out = self.ssm(x_norm)

        # Apply gating if enabled
        if self.gating:
            gate_val = torch.sigmoid(self.gate(x_norm))
            ssm_out = gate_val * ssm_out

        # First residual connection
        x = x + self.dropout(ssm_out)

        # Second normalization
        x_norm = self.norm2(x)

        # MLP layer
        mlp_out = self.mlp(x_norm)

        # Second residual connection
        x = x + mlp_out

        return x


class S5Basecaller(nn.Module):
    def __init__(self, input_dim=1, model_dim=128, num_classes=5, depth=4, d_state=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Create a sequence of S5 blocks
        self.s5 = nn.Sequential(
            *[S5Block(d_model=model_dim, d_state=d_state, dropout=dropout) for _ in range(depth)]
        )

        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):  # x: (B, L, 1)
        x = self.input_proj(x)
        x = self.s5(x)
        logits = self.classifier(x)  # (B, L, num_classes)
        return F.log_softmax(logits, dim=-1)