import torch
import torch.nn as nn
import torch.nn.functional as F


class FouDoRALayer(nn.Module):
    """
    Fourier-based DoRA layer.

    Instead of a low-rank matrix product (LoRA), the weight update lives
    entirely in the low-frequency subspace of the 2-D real FFT of the weight
    matrix.  Only the top-left (freq_h × freq_w) corner of the spectrum is
    learnable; all high-frequency coefficients are held fixed.  The DoRA
    magnitude/direction decomposition is then applied to the reconstructed
    weights, just as in the original DoRA.

    Args:
        base_layer: the nn.Linear layer being adapted.
        n_freqs:    number of low-frequency bins to adapt along each FFT
                    dimension (analogous to rank in LoRA).
        alpha:      scaling factor; the spectral delta is multiplied by
                    alpha / n_freqs before being added.
    """

    def __init__(self, base_layer: nn.Linear, n_freqs: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.n_freqs = n_freqs
        self.scaling = alpha / n_freqs

        out_features, in_features = base_layer.weight.shape
        self.out_features = out_features
        self.in_features = in_features

        # Clamp to valid FFT output dimensions.
        # rfft2 of (out, in) → (out, in//2 + 1) complex coefficients.
        self.freq_h = min(n_freqs, out_features)
        self.freq_w = min(n_freqs, in_features // 2 + 1)

        # Trainable low-frequency spectral deltas, split into real and imag
        # parts so they remain ordinary real-valued parameters.
        # Initialised to zero so the model starts identical to the base.
        self.spectral_delta_real = nn.Parameter(
            torch.zeros(self.freq_h, self.freq_w)
        )
        self.spectral_delta_imag = nn.Parameter(
            torch.zeros(self.freq_h, self.freq_w)
        )

        # DoRA magnitude vector, initialised from row-wise L2 norms of W.
        # Kept in float32 regardless of the base model dtype so gradients are
        # not crushed by bfloat16's limited precision.
        weight = base_layer.weight.data
        self.m = nn.Parameter(weight.norm(p=2, dim=1, keepdim=True).float())

        # Freeze the base-layer parameters.
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def _modified_weight(self) -> torch.Tensor:
        """Return W with the low-frequency spectral delta applied, in float32."""
        W = self.base_layer.weight

        # norm="ortho" makes rfft2/irfft2 an orthonormal pair (Parseval holds):
        # a spectral delta of magnitude d produces a spatial update of magnitude d.
        # The default norm=None has irfft2 divide by N = out*in, so for a 4096×4096
        # layer every spectral update is shrunk by ~1/16M — invisible to the loss.
        W_fft = torch.fft.rfft2(W.detach().float(), norm="ortho")

        fft_h, fft_w = W_fft.shape
        pad_h = fft_h - self.freq_h
        pad_w = fft_w - self.freq_w
        delta_real = F.pad(self.spectral_delta_real, (0, pad_w, 0, pad_h))
        delta_imag = F.pad(self.spectral_delta_imag, (0, pad_w, 0, pad_h))
        delta_full = torch.complex(delta_real, delta_imag) * self.scaling

        # Stay in float32 — casting to bfloat16 here would quantize tiny updates
        # to zero before the DoRA normalisation can use them.
        return torch.fft.irfft2(
            W_fft + delta_full,
            s=(self.out_features, self.in_features),
            norm="ortho",
        )

    def _dora_weights(self) -> torch.Tensor:
        """Apply DoRA direction/magnitude decomposition to the modified weight."""
        W_mod = self._modified_weight()  # float32, no bfloat16 round-trip
        norm = W_mod.norm(p=2, dim=1, keepdim=True)
        direction = W_mod / (norm + 1e-8)
        # Cast to model dtype only here, at the very last step.
        return (self.m * direction).to(self.base_layer.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._dora_weights(), self.base_layer.bias)

    @torch.no_grad()
    def merge_and_unload(self) -> nn.Linear:
        """Merge adapted weights back into the base Linear and return it."""
        W_merged = self._dora_weights()
        self.base_layer.weight.copy_(W_merged)
        return self.base_layer


# ---------------------------------------------------------------------------
# Helpers for applying / removing FouDoRA across a model
# ---------------------------------------------------------------------------

def apply_foudora(
    model: nn.Module,
    n_freqs: int,
    alpha: float = 16.0,
    target_modules: list[str] = ["q_proj", "v_proj"],
) -> nn.Module:
    """Recursively replace target Linear layers with FouDoRALayer."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(model, name, FouDoRALayer(module, n_freqs=n_freqs, alpha=alpha))
        else:
            apply_foudora(module, n_freqs, alpha, target_modules)
    return model


def merge_and_unload_foudora(model: nn.Module) -> nn.Module:
    """Recursively merge all FouDoRALayer instances back into plain Linear layers."""
    for name, module in model.named_children():
        if isinstance(module, FouDoRALayer):
            setattr(model, name, module.merge_and_unload())
        else:
            merge_and_unload_foudora(module)
    return model
