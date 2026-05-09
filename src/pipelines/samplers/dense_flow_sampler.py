"""
Dense Flow Sampler.

Provides diffusion and sampling utilities for dense latent flow matching.
Standard Flow: x_t = (1 - (1 - sigma_min) * t) * x0 + t * noise
Bridge Flow: x_t = (1 - t) * x_target + t * x_source
"""

import torch
from tqdm import tqdm


class DenseFlowSampler:
    """
    Sampler for Dense Latent Flow Matching.

    Implements standard flow matching diffusion and Euler sampling.
    """

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def diffuse(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Diffuse x0 with noise at time t.
        x_t = (1 - t) * x0 + (sigma_min + (1 - sigma_min) * t) * noise
        """
        B = x0.shape[0]
        # Reshape t to match (B, 1, 1) for broadcasting
        t_view = t.view(B, *([1]* len(x0.shape[1:])))
        return (1.0 - t_view) * x0 + (
            self.sigma_min + (1.0 - self.sigma_min) * t_view
        ) * noise

    def bridge_diffuse(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Diffuse from x_source (at t=1) to x_target (at t=0).
        x_t = (1 - t) * x_target + t * x_source
        """
        B = x_target.shape[0]
        t_view = t.view(B, *([1]* len(x_target.shape[1:])))  # Reshape t for broadcasting
        return (1.0 - t_view) * x_target + t_view * x_source

    def get_v(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Velocity target for standard flow matching.
        dx/dt = -x0 + (1 - sigma_min) * noise
        """
        return (1.0 - self.sigma_min) * noise - x0

    def bridge_get_v(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Velocity target for bridge flow.
        dx/dt = x_source - x_target
        """
        return x_source - x_target

    def compute_v_pred(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_source: torch.Tensor | None = None,
        predict_x: bool = False,
    ) -> torch.Tensor:
        """
        Compute predicted velocity from model output.

        Args:
            model_output: Model prediction (v or x0)
            x_t: Current latent
            t: Current time [0, 1] (B,)
            x_source: Source latent for bridge flow
            predict_x: If True, model_output is predicted x0
        """
        if not predict_x:
            return model_output

        B = x_t.shape[0]
        # Reshape t to match (B, 1, 1) for broadcasting
        t_view = t.view(B, 1, 1)

        if x_source is not None:
            # Bridge flow: v = x_source - x0
            return x_source - model_output
        else:
            # Standard flow: v = (1-sigma_min)*noise - x0
            # x_t = (1-t)*x0 + (sigma_min + (1-sigma_min)*t)*noise
            # noise = (x_t - (1-t)*x0) / (sigma_min + (1-sigma_min)*t)
            noise_coeff = self.sigma_min + (1.0 - self.sigma_min) * t_view
            noise_est = (x_t - (1.0 - t_view) * model_output) / noise_coeff.clamp(
                min=1e-5
            )
            return (1.0 - self.sigma_min) * noise_est - model_output

    def compute_x0_pred(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_source: torch.Tensor = None,
        predict_x: bool = False,
    ) -> torch.Tensor:
        """
        Compute predicted x0 from model output.

        Args:
            model_output: Model prediction (v or x0)
            x_t: Current latent
            t: Current time [0, 1] (B,)
            x_source: Source latent for bridge flow
            predict_x: If True, model_output is predicted x0
        """
        if predict_x:
            return model_output

        B = x_t.shape[0]
        t_view = t.view(B, 1, 1)

        if x_source is not None:
            # Bridge flow: v = x_source - x0 => x0 = x_source - v
            return x_source - model_output
        else:
            # Standard flow: x0 = (1-sigma_min)*x_t - (sigma_min + (1-sigma_min)*t)*v
            noise_coeff = self.sigma_min + (1.0 - self.sigma_min) * t_view
            return (1.0 - self.sigma_min) * x_t - noise_coeff * model_output

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        cond: torch.Tensor | None,
        x_source: torch.Tensor = None,
        steps: int = 50,
        verbose: bool = False,
        predict_x: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples from noise/source (t=1) to target (t=0).

        For standard flow: starts from noise at t=1, integrates to t=0
        For bridge flow: starts from x_source at t=1, integrates to t=0

        Args:
            model: Denoiser model
            noise: Starting latent (B, L, C) - used for standard flow
            cond: Conditioning tensor (B, L_cond, C_cond)
            x_source: Source latent for bridge flow (replaces noise if provided)
            steps: Number of Euler steps
            verbose: Show progress bar
            predict_x: If True, model predicts x0 instead of velocity

        Returns:
            x_0: Refined latent at t=0
        """
        device = noise.device
        dt = 1.0 / steps

        # For bridge flow, start from x_source; for standard flow, start from noise
        if x_source is not None:
            x_t = x_source.clone()  # Bridge: start from x_source at t=1
        else:
            x_t = noise  # Standard: start from noise at t=1
        B = len(x_t)

        indices = range(steps)
        if verbose:
            indices = tqdm(indices, desc="Sampling")

        for i in indices:
            t_val = 1.0 - i * dt
            # Scale t to [0, 1000] for model consistency
            t_model = torch.full((B,), t_val * 1000.0, device=device)

            model_output = model(x_t, t_model, cond)

            v_pred = self.compute_v_pred(
                model_output,
                x_t,
                torch.full((B,), t_val, device=device),
                x_source=x_source,
                predict_x=predict_x,
            )

            # Both standard and bridge flow: dx/dt = v_pred
            # Backward Euler: x_{t-dt} = x_t - v_pred * dt
            x_t = x_t - v_pred * dt

        return x_t
