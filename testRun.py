import contextlib
import json
import logging
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch.utils.checkpoint import checkpoint

from KolSol.kolsol.torch.solver import KolSol

try:
    from tqdm import trange
except ImportError:
    trange = None

main_path = os.getcwd()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
torch.manual_seed(999)
np.random.seed(999)

# Simulation parameters
nk = 64
nf = 4
re = 40.0
dt = 0.0025
Topt = 100
control_scale = 4
beta_value = 0.0
state_dtype = torch.complex64
measure_field = True
mean_epsilon, std_epsilon = 0.11726184557297091, 0.03806973962448417  # dissipation mean and std of the training data
nr_dev = 2

ic_file = f"{main_path}/kolmogorov_flow_re40/datasets/fourier_128x128_IC_control.h5"  # the evaluation's starting state

show_progress = True
progress_every = 1000   # steps between progress-bar updates; each update makes the CPU wait for the GPU
use_cuda_graph = True   # replay a recording of each control interval (GPU only, and only when no derivatives are tracked)


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


# Single step RK4 as per "control_optimization\control_law_evaluation.py"
def rk4_step(ks, u_hat, beta, dt):
    k1 = ks.dynamics_control_betaSinVar(u_hat, beta)
    beta1 = beta2 = beta3 = beta

    u_hat_1 = u_hat + 0.5 * dt * k1
    k2 = ks.dynamics_control_betaSinVar(u_hat_1, beta1)

    u_hat_2 = u_hat + 0.5 * dt * k2
    k3 = ks.dynamics_control_betaSinVar(u_hat_2, beta2)

    u_hat_3 = u_hat + dt * k3
    k4 = ks.dynamics_control_betaSinVar(u_hat_3, beta3)

    return (u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)).to(u_hat.dtype)  # keep the state's dtype, like the in-place +=


# True if any value carries a derivative: requires_grad (reverse mode) or a dual tangent (forward mode)
def tracks_derivatives(*values):
    return any(torch.is_tensor(v) and (v.requires_grad or fwAD.unpack_dual(v).tangent is not None) for v in values)


# One pass of the evaluation's loop body: measure the flow, then take n_steps RK4 steps
def control_interval(ks, u_hat, beta, n_steps=control_scale):
    # the solver calls the evaluation makes every control step (its lines 177 and 187)
    if measure_field:
        u_field = ks.fourier_to_phys(u_hat, nref=2 * nk).to(dtype=torch.float32)  # noqa: F841 (encoder input)
    eps = ks.dissip(u_hat).to(dtype=torch.float32)
    for _ in range(n_steps):
        u_hat = rk4_step(ks, u_hat, beta, dt)
    return u_hat, eps


# Record one control interval as a CUDA graph: replaying it re-runs the same GPU jobs without the CPU preparing each one
def record_control_interval(ks, u_hat0, beta):
    u_buf = u_hat0.clone()  # the recording always reads and writes these same two buffers
    eps_buf = torch.zeros((), dtype=torch.float32, device=device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up runs, so the FFT plans exist before recording
        for _ in range(3):
            control_interval(ks, u_buf, beta)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        u_next, eps = control_interval(ks, u_buf, beta)
        u_buf.copy_(u_next)
        eps_buf.copy_(eps)
    return graph, u_buf, eps_buf

# Time loop but without a controller
def run(ks, u_hat0, beta, Nt):
    ext_threshold = mean_epsilon + nr_dev * std_epsilon
    n_intervals, n_rest = divmod(Nt, control_scale)
    derivatives = tracks_derivatives(u_hat0, beta)
    reverse_mode = torch.is_grad_enabled() and any(torch.is_tensor(v) and v.requires_grad for v in (u_hat0, beta))
    forward_mode = any(torch.is_tensor(v) and fwAD.unpack_dual(v).tangent is not None for v in (u_hat0, beta))
    use_checkpoint = reverse_mode and not forward_mode  # PyTorch's checkpoint cannot carry forward-mode tangents
    use_graph = use_cuda_graph and device == 'cuda' and not derivatives
    report_every = max(1, progress_every // control_scale)
    eps_vals = []
    u_hat = u_hat0.clone()

    with contextlib.nullcontext() if derivatives else torch.no_grad():
        if use_graph:
            graph, u_buf, eps_buf = record_control_interval(ks, u_hat, beta)
        steps = trange(n_intervals, unit_scale=control_scale) if (trange is not None and show_progress) else range(n_intervals)
        for i in steps:
            if use_graph:
                graph.replay()
                eps = eps_buf.clone()  # a copy on the GPU: nothing waits
            elif use_checkpoint:  # keep only each interval's starting state; the rest is recomputed in the backward pass
                u_hat, eps = checkpoint(control_interval, ks, u_hat, beta, use_reentrant=False)
            else:
                u_hat, eps = control_interval(ks, u_hat, beta)
            eps_vals.append(eps)
            if hasattr(steps, "set_postfix") and i % report_every == 0:  # .item() makes the CPU wait for the GPU
                steps.set_postfix({"eps": f"{eps.item():.4f}", "beta": f"{float(beta):.4f}"})
        if use_graph:
            u_hat = u_buf.clone()
        if n_rest:  # a last, shorter interval when Nt is not a multiple of control_scale
            u_hat, eps = control_interval(ks, u_hat, beta, n_rest)
            eps_vals.append(eps)

    eps_vals = torch.stack(eps_vals)
    states_type = torch.where(eps_vals >= ext_threshold, 2, 0)  # without the clusters only "extreme" (2) can be identified
    num_ext_events = int((states_type == 2).sum())             # the loop's only other wait for the GPU, after the run
    return u_hat, eps_vals, states_type, num_ext_events

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    Nt = int(Topt / dt)  # number of integration timesteps

    sync()
    t0 = time.perf_counter()
    ks = KolSol(nk=nk, nf=nf, re=re, ndim=2, device=device)
    sync()
    t_setup = time.perf_counter() - t0

    with h5py.File(ic_file, "r") as f:
        u_hat = f["velocity_train"][:]
    u_hat0 = torch.tensor(u_hat).detach().clone().to(device=device, dtype=state_dtype)

    run_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_T{Topt}s_uncontrolled"
    output_folder = f"{main_path}/output/uncontrolled/{run_name}"
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"{run_name}: N = {2 * nk}, Re = {re}, dt = {dt}, {Nt} steps, beta = {beta_value}, {device}")

    sync()
    t0 = time.perf_counter()
    u_hat_final, eps_vals, states_type, num_ext_events = run(ks, u_hat0, beta_value, Nt)
    sync()
    t_run = time.perf_counter() - t0

    eps_vals = eps_vals.detach().double().cpu().numpy()  # saved as before: float64 copies of the float32 values
    states_type = states_type.cpu().numpy().astype(int)
    betas = np.full(len(eps_vals), float(beta_value))

    with torch.no_grad():
        rhs0 = ks.dynamics_control_betaSinVar(u_hat0, beta_value)
        final_eps = ks.dissip(u_hat_final).item()

    logging.info(f"Extreme Events: {num_ext_events / Nt * 100} % (computed as control_law_evaluation.py does; "
                 f"{100 * num_ext_events / len(eps_vals):.3f} % of the measured states)")
    logging.info(f"Time: solver set-up {t_setup:.3f} s, run {t_run:.2f} s, {1000 * t_run / Nt:.3f} ms per step")

    np.save(f'{output_folder}/betas.npy', betas)
    np.save(f'{output_folder}/eps_vals.npy', eps_vals)
    np.save(f'{output_folder}/states_type.npy', states_type)
    np.save(f'{output_folder}/u_hat_final.npy', u_hat_final.detach().cpu().numpy())

    info = {
        "parameters": {"N": 2 * nk, "nk": nk, "nf": nf, "re": re, "dt": dt, "Topt": Topt, "steps": Nt,
                       "control_scale": control_scale, "beta": beta_value, "state_dtype": str(state_dtype),
                       "measure_field": measure_field, "use_cuda_graph": use_cuda_graph, "device": device,
                       "device_name": torch.cuda.get_device_name() if device == 'cuda' else "cpu",
                       "torch": str(torch.__version__)},
        "timing_s": {"solver_setup": t_setup, "run": t_run, "per_step": t_run / Nt},
        "reference_values": {"rhs_norm_t0": rhs0.abs().pow(2).sum().sqrt().item(),
                             "eps_mean": float(eps_vals.mean()), "eps_std": float(eps_vals.std()),
                             "eps_max": float(eps_vals.max()), "eps_final": final_eps,
                             "state_norm_final": u_hat_final.abs().pow(2).sum().sqrt().item()},
    }

    with open(f'{output_folder}/run_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    logging.info(f"Finished the run: {run_name}. Data saved to {output_folder}")


if __name__ == "__main__":
    main()