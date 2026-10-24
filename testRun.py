import json
import logging
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch

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

# Time loop but without a controller
def run(ks, u_hat0, beta, Nt):
    ext_threshold = mean_epsilon + nr_dev * std_epsilon
    betas, eps_vals, states_type = [], [], []
    num_ext_events = 0
    u_hat = u_hat0.clone()

    with torch.no_grad():
        steps = trange(Nt) if (trange is not None and show_progress) else range(Nt)
        for step in steps:
            if step % control_scale == 0:
                # the solver calls the evaluation makes every control step (its lines 177 and 187)
                if measure_field:
                    u_field = ks.fourier_to_phys(u_hat, nref=2 * nk).to(dtype=torch.float32)  # noqa: F841 (encoder input)
                eps = ks.dissip(u_hat).to(dtype=torch.float32)
                z_type = 2 if eps >= ext_threshold else 0  # without the clusters only "extreme" (2) can be identified

                eps_vals.append(eps.item())
                states_type.append(z_type)
                betas.append(float(beta))
                if hasattr(steps, "set_postfix"):
                    steps.set_postfix({"eps": f"{eps.item():.4f}", "beta": f"{float(beta):.4f}"})
                if z_type == 2:
                    num_ext_events += 1

            u_hat = rk4_step(ks, u_hat, beta, dt)

    return u_hat, np.array(betas), np.array(eps_vals), np.array(states_type), num_ext_events

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
    u_hat_final, betas, eps_vals, states_type, num_ext_events = run(ks, u_hat0, beta_value, Nt)
    sync()
    t_run = time.perf_counter() - t0

    with torch.no_grad():
        rhs0 = ks.dynamics_control_betaSinVar(u_hat0, beta_value)
        final_eps = ks.dissip(u_hat_final).item()

    logging.info(f"Extreme Events: {num_ext_events / Nt * 100} % (computed as control_law_evaluation.py does; "
                 f"{100 * num_ext_events / len(eps_vals):.3f} % of the measured states)")
    logging.info(f"Time: solver set-up {t_setup:.3f} s, run {t_run:.2f} s, {1000 * t_run / Nt:.3f} ms per step")

    np.save(f'{output_folder}/betas.npy', betas)
    np.save(f'{output_folder}/eps_vals.npy', eps_vals)
    np.save(f'{output_folder}/states_type.npy', states_type)
    np.save(f'{output_folder}/u_hat_final.npy', u_hat_final.cpu().numpy())

    info = {
        "parameters": {"N": 2 * nk, "nk": nk, "nf": nf, "re": re, "dt": dt, "Topt": Topt, "steps": Nt,
                       "control_scale": control_scale, "beta": beta_value, "state_dtype": str(state_dtype),
                       "measure_field": measure_field, "device": device,
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