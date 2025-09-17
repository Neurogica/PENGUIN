from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from models.layers.S5_init import *
from models.layers.S5_jax_compat import associative_scan

# Runtime functions


@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j
    # return A_j * A_i, A_j * Bu_i + Bu_j
    return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)


def apply_ssm(Lambda_bars: torch.Tensor, B_bars, C_tilde, D, input_sequence, state=None, bidir: bool = False):
    cinput_sequence = input_sequence.type(Lambda_bars.dtype)  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Zero-pad for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    if state is not None:
        # Bu_elements = torch.cat(((state).unsqueeze(0), Bu_elements), dim=0)
        # Lambda_bars = torch.cat((torch.ones_like(state.unsqueeze(0)), Lambda_bars), dim=0)
        # Manually compute first step (Lambda_bar=1 so no change)
        Bu_elements[0] = Bu_elements[0] + state * Lambda_bars[0]

    _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(binary_operator, (Lambda_bars, Bu_elements), reverse=True)
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1]  # torch.stack((_[-1], xs[-1]))


def apply_ssm_liquid(Lambda_bars, B_bars, C_tilde, D, input_sequence, state=None, bidir: bool = False):
    """Liquid time constant SSM \u00e1 la dynamical systems given in Eq. 8 of
    https://arxiv.org/abs/2209.12951"""
    cinput_sequence = input_sequence.type(Lambda_bars.dtype)  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Zero-pad for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    if state is not None:
        # Manually compute first step (Lambda_bar=1 so no change)
        Bu_elements[0] = Bu_elements[0] + state * Lambda_bars[0]

    _, xs = associative_scan(binary_operator, (Lambda_bars + Bu_elements, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(binary_operator, (Lambda_bars, Bu_elements), reverse=True)
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1]


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)  # (replaced by -1)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


Initialization = Literal["dense_columns", "dense", "factorized"]


class S5SSM(torch.nn.Module):
    def __init__(
        self,
        lambdaInit: torch.Tensor,
        V: torch.Tensor,
        Vinv: torch.Tensor,
        h: int,
        p: int,
        dt_min: float,
        dt_max: float,
        liquid: bool = False,
        factor_rank: Optional[int] = None,
        discretization: Literal["zoh", "bilinear"] = "zoh",
        bcInit: Initialization = "factorized",
        degree: int = 1,
        bidir: bool = False,
    ):
        """The S5 SSM
        Args:
            lambdaInit  (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            h           (int32):     Number of features of input seq
            p           (int32):     state size
            k           (int32):     rank of low-rank factorization (if used)
            bcInit      (string):    Specifies How B and C are initialized
                        Options: [factorized: low-rank factorization,
                                dense: dense matrix drawn from Lecun_normal]
                                dense_columns: dense matrix where the columns
                                of B and the rows of C are each drawn from Lecun_normal
                                separately (i.e. different fan-in then the dense option).
                                We found this initialization to be helpful for Pathx.
            discretization: (string) Specifies discretization method
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
            liquid:         (bool): use liquid_ssm from LiquidS4
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.Lambda = torch.nn.Parameter(lambdaInit)
        self.degree = degree
        self.liquid = liquid
        self.bcInit = bcInit
        self.bidir = bidir
        # TODO:
        # if self.clip_eigs:
        #    self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im

        # the P-dim of C can needs to be 2P for bidir
        cp = p
        if self.bidir:
            cp *= 2

        if bcInit == "complex_normal":
            self.C = torch.nn.Parameter(torch.normal(0, 0.5**0.5, (h, cp), dtype=torch.complex64))
            self.B = torch.nn.Parameter(init_VinvB(lecun_normal(), Vinv)((p, h), torch.float))
        elif (bcInit == "dense_columns") or (bcInit == "dense"):
            if bcInit == "dense_columns":
                B_eigen_init = init_columnwise_VinvB
                B_init = init_columnwise_B
                C_init = init_rowwise_C
            elif bcInit == "dense":
                B_eigen_init = init_VinvB
                B_init = C_init = lecun_normal()
            # TODO: make init_*VinvB all a the same interface
            self.B = torch.nn.Parameter(B_eigen_init(B_init, Vinv)((p, h), torch.float))
            if self.bidir:
                C = torch.cat([init_CV(C_init, (h, p), V), init_CV(C_init, (h, p), V)], axis=-1)
            else:
                C = init_CV(C_init, (h, p), V)
            self.C = torch.nn.Parameter(C)
        elif bcInit == "factorized":
            print("[WARN]: factorized was removed from the original repo, might be for a reason :?")
            # Use a low rank factorization of rank k for B and C
            self.BH = torch.nn.Parameter(as_complex(init_columnwise_B((h, k, 2), torch.float32)))
            self.BP = torch.nn.Parameter(as_complex(init_columnwise_B((p, k, 2), torch.float32)))
            self.CH = torch.nn.Parameter(as_complex(init_rowwise_C((k, h, 2), torch.float32)))
            self.CP = torch.nn.Parameter(as_complex(init_rowwise_C((k, cp, 2), torch.float32)))
            # self.BH = torch.nn.Parameter(init_columnwise_B((h, k), torch.complex64))
            # self.BP = torch.nn.Parameter(init_columnwise_B((p, k), torch.complex64))
            # self.CH = torch.nn.Parameter(init_rowwise_C((k, h), torch.complex64))
            # self.CP = torch.nn.Parameter(init_rowwise_C((k, p), torch.complex64))
        else:
            raise NotImplementedError(f"BC_init method {bcInit} not implemented")

        # Initialize feedthrough (D) matrix
        self.D = torch.nn.Parameter(
            torch.rand(
                h,
            )
        )
        self.log_step = torch.nn.Parameter(init_log_steps(p, dt_min, dt_max))
        if discretization == "zoh":
            self.discretize = discretize_zoh
        elif discretization == "bilinear":
            self.discretize = discretize_bilinear
        else:
            raise ValueError(f"Unknown discretization {discretization}")

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C_tilde.shape[-2]))

    def get_BC_tilde(self):
        if self.bcInit in ["dense_columns", "dense", "complex_normal"]:
            B_tilde = as_complex(self.B)
            C_tilde = self.C
        elif self.bcInit == "factorized":
            B_tilde = self.BP @ self.BH.T
            C_tilde = self.CH.T @ self.CP
        return B_tilde, C_tilde

    def forward_rnn(self, signal, prev_state, step_scale=1.0):
        assert not self.bidir, "Can't use bidirectional when manually stepping"
        B_tilde, C_tilde = self.get_BC_tilde()
        step = step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bar = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert B_bar.shape[-2] == B_bar.shape[-1], "higher-order input operators must be full-rank"
            B_bar **= self.degree

        # https://arxiv.org/abs/2209.12951v1, Eq. 9
        Bu = B_bar @ signal.type(B_bar.dtype)
        if self.liquid:
            Lambda_bar += Bu
        # https://arxiv.org/abs/2208.04933v2, Eq. 2
        x = Lambda_bar * prev_state + Bu
        y = (C_tilde @ x + self.D * signal).real
        return y, x

    def forward(self, signal, step_scale=1.0, state=None, return_state=False):
        B_tilde, C_tilde = self.get_BC_tilde()

        if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
            step = step_scale * torch.exp(self.log_step)
        else:
            # TODO: This is very expensive due to individual steps being multiplied by B_tilde in self.discretize
            step = step_scale[:, None] * torch.exp(self.log_step)

        # print(f'{self.Lambda.shape=} {B_tilde.shape=} {step.shape=}')
        # Lambda_bars, B_bars = torch.vmap(lambda s: self.discretize(self.Lambda, B_tilde, s))(step)
        # print(Lambda_bars.shape, B_bars.shape)
        Lambda_bars, B_bars = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert B_bars.shape[-2] == B_bars.shape[-1], "higher-order input operators must be full-rank"
            B_bars **= self.degree

        assert not (self.bidir and (state is not None)), "injecting state is not compatible with bidirectional S5"

        forward = apply_ssm_liquid if self.liquid else apply_ssm
        out, state = forward(Lambda_bars, B_bars, C_tilde, self.D, signal, state=state, bidir=self.bidir)
        # NOTE: technically it could work in a limited sense; taking the first and last element
        #   but that wouldn't be equivalent to running bidir on full sequences.
        #  It would be more like a circular S5 where you keep splicing the new signal into it;
        #   we leave implementing/testing this as an exercise to the reader
        assert not (self.bidir and return_state), "return_state does not work with bidirectional S5"
        if return_state:
            return out, state
        return out


class S5(torch.nn.Module):
    def __init__(
        self,
        width: int,
        state_width: Optional[int] = None,
        factor_rank: Optional[int] = None,
        block_count: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        liquid: bool = False,
        degree: int = 1,
        bidir: bool = False,
        bcInit: Optional[Initialization] = None,
    ):
        super().__init__()
        state_width = state_width or width
        assert state_width % block_count == 0, "block_count should be a factor of state_width"

        block_size = state_width // block_count
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        Vinv = V.conj().T
        Lambda, B, V, B_orig, Vinv = map(lambda v: torch.tensor(v, dtype=torch.complex64), (Lambda, B, V, B_orig, Vinv))
        if block_count > 1:
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Lambda = (Lambda * torch.ones((block_count, block_size))).ravel()
            V = torch.block_diag(*([V] * block_count))
            Vinv = torch.block_diag(*([Vinv] * block_count))

        assert bool(factor_rank) != bool(bcInit != "factorized"), "Can't have `bcInit != factorized` and `factor_rank` defined"
        bc_init = "factorized" if factor_rank is not None else (bcInit or "dense")
        self.width = width
        self.seq = S5SSM(
            Lambda, V, Vinv, width, state_width, dt_min, dt_max, factor_rank=factor_rank, bcInit=bc_init, liquid=liquid, degree=degree, bidir=bidir
        )

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal, step_scale=1.0, state=None, return_state=False):
        # NOTE: step_scale can be float | Tensor[batch] | Tensor[batch, seq]
        if not torch.is_tensor(step_scale):
            # Duplicate across batchdim
            step_scale = torch.ones(signal.shape[0], device=signal.device) * step_scale

        if state is None:
            return torch.vmap(lambda s, ss: self.seq(s, step_scale=ss, return_state=return_state))(signal, step_scale)
        else:
            return torch.vmap(lambda s, ss, _state: self.seq(s, step_scale=ss, state=_state, return_state=return_state))(signal, step_scale, state)
