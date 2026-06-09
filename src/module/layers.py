"""Graph VAE layers and submodules."""

import math
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.init as init


type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor

DEBUG_MODE = (
    any(module in sys.modules for module in ("pydevd", "debugpy", "pdb"))
    or sys.gettrace() is not None
)


class LinearLayer(nn.Module):
    """A linear layer with optional batch normalization and activation."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        use_batch_norm: bool = False,
        act: nn.Module | None = None,
        truncated_init: bool = False,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.use_batch_norm = use_batch_norm
        self.act = act
        self.truncated_init = truncated_init
        self.fc = nn.Linear(n_inputs, n_outputs, bias=True)
        self._init_weights()
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(n_inputs, track_running_stats=True)

        if DEBUG_MODE:
            warnings.filterwarnings(
                "ignore", message=".*Full backward hook is firing.*"
            )
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_batch_norm:
            x = x.permute(0, 2, 1)
            x = self.bn(x)
            x = x.permute(0, 2, 1)

        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)

        return x

    def _init_weights(self) -> None:
        """Initialize the linear layer's weights based on the activation."""

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc.weight)
        if self.truncated_init:
            init.trunc_normal_(self.fc.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        elif self.act is None:
            init.xavier_normal_(self.fc.weight, gain=1.0)
        elif isinstance(self.act, nn.Hardtanh):
            init.xavier_normal_(self.fc.weight, gain=nn.init.calculate_gain("tanh"))
        elif isinstance(self.act, nn.ReLU):
            init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        elif isinstance(self.act, nn.LeakyReLU):
            init.kaiming_normal_(
                self.fc.weight, a=self.act.negative_slope, nonlinearity="leaky_relu"
            )
        elif isinstance(self.act, nn.GELU):
            init.xavier_normal_(self.fc.weight, gain=1.0)
        elif isinstance(self.act, nn.ELU):
            std = math.sqrt(1.55 / fan_in)
            init.normal_(self.fc.weight, mean=0.0, std=std)
        else:
            init.xavier_normal_(self.fc.weight, gain=1.0)

        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        return


class LapResNet(nn.Module):
    """Residual block utilizing Laplacian operator."""

    def __init__(self, n_outputs: int, act: nn.Module) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.act = act
        self.bn_fc0 = LinearLayer(
            2 * n_outputs, n_outputs, use_batch_norm=True, act=act
        )
        self.bn_fc1 = LinearLayer(
            2 * n_outputs,
            n_outputs,
            use_batch_norm=True,
            act=act,
            truncated_init=True,
        )
        return

    def _apply_L(self, L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch, node, feat = x.size()
        if L.size(0) == node:
            x_flat = x.permute(1, 0, 2).reshape(node, batch * feat)
            return L.mm(x_flat).reshape(node, batch, feat).permute(1, 0, 2)

        return L.mm(x.view(-1, feat)).view(batch, node, feat)

    def forward(self, L: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        L_x = self._apply_L(L, x)
        x = torch.cat([x, L_x], 2)
        x = self.bn_fc0(x)
        L_x = self._apply_L(L, x)
        x = torch.cat([x, L_x], 2)
        x = self.bn_fc1(x)
        return x + inputs


class DirResNet(nn.Module):
    """Residual block utilizing Dirac operators."""

    def __init__(self, n_outputs: int, act: nn.Module, res_f: bool = False) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.bn_fc0 = LinearLayer(
            2 * n_outputs, n_outputs, use_batch_norm=True, act=act
        )
        self.bn_fc1 = LinearLayer(
            2 * n_outputs,
            n_outputs,
            use_batch_norm=True,
            act=act,
            truncated_init=True,
        )
        self.res_f = res_f
        return

    def _apply_Di(
        self, Di: torch.Tensor, v: torch.Tensor, n_faces: int
    ) -> torch.Tensor:
        batch, n_nodes, n_inputs = v.size()
        feat4 = n_inputs // 4
        if Di.size(1) == n_nodes * 4:
            x = v.reshape(batch, n_nodes * 4, feat4).permute(1, 0, 2)
            x = Di.mm(x.reshape(n_nodes * 4, batch * feat4))
            return (
                x.reshape(n_faces * 4, batch, feat4)
                .permute(1, 0, 2)
                .reshape(batch, n_faces, n_inputs)
            )

        x = v.view(batch * n_nodes * 4, feat4)
        return Di.mm(x).view(batch, n_faces, n_inputs)

    def _apply_DiA(
        self, DiA: torch.Tensor, f: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        batch, n_faces, n_inputs = f.size()
        feat4 = n_inputs // 4
        if DiA.size(1) == n_faces * 4:
            x = f.reshape(batch, n_faces * 4, feat4).permute(1, 0, 2)
            x = DiA.mm(x.reshape(n_faces * 4, batch * feat4))
            return (
                x.reshape(n_nodes * 4, batch, feat4)
                .permute(1, 0, 2)
                .reshape(batch, n_nodes, n_inputs)
            )

        x = f.view(batch * n_faces * 4, feat4)
        return DiA.mm(x).view(batch, n_nodes, n_inputs)

    def forward(
        self, Di: torch.Tensor, DiA: torch.Tensor, v: torch.Tensor, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_nodes, n_inputs = v.size()
        _, n_faces, _ = f.size()
        x = self._apply_Di(Di, v, n_faces)
        x = torch.cat([f, x], 2)
        x = self.bn_fc0(x)
        f_out = x
        x = self._apply_DiA(DiA, x, n_nodes)
        x = torch.cat([v, x], 2)
        x = self.bn_fc1(x)
        v_out = x
        return v + v_out, f_out


class PointwiseResNet(nn.Module):
    """Pointwise residual block (no graph/mesh operators)."""

    def __init__(self, n_outputs: int, act: nn.Module) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.act = act
        self.bn_fc0 = LinearLayer(
            n_outputs, n_outputs, use_batch_norm=True, act=act
        )
        self.bn_fc1 = LinearLayer(
            n_outputs,
            n_outputs,
            use_batch_norm=True,
            act=act,
            truncated_init=True,
        )
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.bn_fc0(inputs)
        x = self.bn_fc1(x)
        return x + inputs


def global_average(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling over vertices."""
    return torch.mean(x, dim=1)


def debug_check(
    _not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t
) -> None:
    if isinstance(tensor_out, tuple):
        tensor = tensor_out[0]
    else:
        tensor = tensor_out

    if torch.any(torch.isnan(tensor)):
        breakpoint()
    elif torch.any(torch.isinf(tensor)):
        breakpoint()

    return None
