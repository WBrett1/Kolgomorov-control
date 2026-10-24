import numpy as np
import sys
import h5py
import os
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import logging
import numpy as np
from tqdm import trange
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from autoencoders.symmetry_aware_autoencoder import DiscreteSymmetries, SymAutoencoder, STN
from autoencoders.autoencoder import Autoencoder
from KolSol.kolsol.torch.solver import KolSol
from utils import flatten_clusters, get_num_clusters_each, compute_Ji, identify_point_type, Actuator, load_clusters

main_path = os.getcwd()
requires_grad = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(999)
np.random.seed(999)

# Optimization parameters ========================
autoencoder_type = "sae"        # "sae": symmetry-aware autoencoder or "cae": autoencoder
Topt = 10000  # length of the simulation in seconds

mean_epsilon, std_epsilon = 0.11726184557297091, 0.03806973962448417  # energy dissipation rate mean and std from entire dataset

# latent space normalization params based on the dataset used for clustering
if autoencoder_type == "sae":
    prev_mean = torch.tensor([[  3.0741701126098633, -2.3254714012145996, 3.2388222217559814, -0.9291426539421082, -36.08496856689453, 3.460550308227539, 6.827239990234375, 18.972339630126953, 0.11595899091001875]], device=device, dtype=torch.float32) 
    prev_std = torch.tensor([[ 6.697792053222656, 6.32340145111084, 5.581307888031006, 7.944993019104004, 6.816481590270996, 5.449934482574463, 7.902254104614258, 6.43773078918457, 0.0375438561795101 ]], device=device, dtype=torch.float32)
else:
    prev_mean = torch.tensor([[  -1.3589740991592407, -0.885854959487915, 3.369107723236084, -2.049339771270752, -3.06653094291687, 1.381832480430603, 5.244030952453613, -1.4049558639526367, 0.11595899091001875]], device=device, dtype=torch.float32) 
    prev_std = torch.tensor([[ 25.471277236938477, 7.641894817352295, 18.170530319213867, 10.358161926269531, 7.718150615692139, 26.935338973999023, 22.47753143310547, 11.527128219604492, 0.0375438561795101 ]], device=device, dtype=torch.float32)

nr_dev = 2                  # number of std deviations above the mean to consider an extreme event

output_size = (32, 32)      # interpolation output, used as input to the autoencoder

max_beta = 0.8              # max value for the control signal

# Define the optimized control signal beta
if autoencoder_type == "sae":
    # Current optimum values for SA-nCAE encoding
    bk = torch.tensor([-0.2609, -0.4042, -0.1078,  0.1709, -0.2596, -0.2739,  0.0239, -0.4088,
            0.2458, -0.2786, -0.2565, -0.3356,  0.1322, -0.1044, -0.1982, -0.4617,
            -0.2593, -0.3200, -0.2154, -0.0839, -0.4329,  0.0739, -0.0184, -0.1600,
            -0.2018, -0.1380,  0.1210, -0.3918, -0.1444, -0.3558,  0.1191, -0.1722,
            -0.3929, -0.0658,  0.3081, -0.2657, -0.2719,  0.0552, -0.1476,  0.1891,
            -0.2475, -0.2284, -0.2673, -0.0038,  0.2129, -0.2454,  0.2698, -0.2627,
            -0.1533, -0.5148, -0.4099, -0.2197, -0.1180, -0.2552, -0.0197, -0.3137,
            0.3119,  0.1702, -0.4818,  0.1747,  0.4014,  0.4168, -0.2226,  0.0833,
            0.3049,  0.2311,  0.5693,  0.3661,  0.0575,  0.1820,  0.4164,  0.5130,
            0.3941,  0.4216,  0.5556,  0.3269,  0.3839,  0.6345,  0.1091,  0.4141])
else:
    # Current optimum values for nCAE encoding
    bk = torch.tensor([-3.5360e-01, -4.6725e-01,  9.9112e-01,  1.0564e+00, -1.4561e+00,
         1.2048e-01, -9.4817e-01,  1.0073e+00,  4.9794e-01, -1.1298e+00,
         4.3566e-01, -9.1907e-02,  1.6728e-01, -8.3564e-01,  4.0503e-01,
         5.7219e-01, -7.4552e-01, -1.4794e+00,  1.1944e+00, -4.9208e-01,
        -2.9938e-03,  1.5234e-01, -6.2598e-01, -9.4694e-01, -4.8437e-01,
        -4.3874e-02,  3.1132e-01,  1.2524e+00,  1.2082e-01, -9.2136e-01,
        -5.6643e-01, -1.2285e-02,  5.0090e-01,  4.0321e-01, -8.4903e-02,
        -8.6916e-01, -1.3425e-01, -2.5631e-01, -7.7391e-01,  1.8403e-01,
         1.6095e-01, -2.6206e-01,  1.0028e+00,  7.4287e-01,  2.4040e-01,
        -4.2203e-01, -5.7964e-01, -6.0212e-02, -1.1775e+00, -4.1092e-01,
        -2.7194e-01, -1.1223e+00, -1.4424e+00, -1.2222e+00, -3.5351e-01,
        -5.8227e-01,  9.5897e-01,  1.2356e+00,  6.9212e-01,  2.0171e+00,
         3.4017e-01, -6.0846e-01, -7.5259e-02,  9.2454e-01, -9.1156e-01,
         1.8396e+00, -1.2317e-01,  2.9908e+00,  1.2563e+00,  6.5800e-01,
         1.6222e+00,  1.4071e+00,  3.9590e+00,  1.3533e+00,  1.4532e+00,
         3.9484e+00,  2.6436e+00,  2.2705e+00])
# END Optimization parameters ========================


# Setup logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# =================================
# Load SAE/CAE autoencoder
z_size = 8      # latent space size
if autoencoder_type == "sae":
    state_dict = torch.load(f"{main_path}/autoencoders/pretrained_models/sae_autoencoder_{z_size}.pth", map_location=device)
    stn = STN((2,32,32))
    discrete_symmetries = DiscreteSymmetries().to(device=device)
    autoencoder_base = Autoencoder(latent_dim=z_size, base=32, activation=nn.Tanh()).to(device=device)
    autoencoder = SymAutoencoder(autoencoder_base, stn, discrete_symmetries).to(device=device)
else:
    state_dict = torch.load(f"{main_path}/autoencoders/pretrained_models/autoencoder_{z_size}.pth", map_location=device)
    autoencoder = Autoencoder(latent_dim=z_size, base=32, activation=nn.Tanh()).to(device=device)

autoencoder.load_state_dict(state_dict)
autoencoder.eval()

# =================================
# Load clustering
# 1.b. Load clustering
clustering_name = 'clusters_sae' if autoencoder_type == "sae" else 'clusters_vcae'
MO_DATA_DIR    = f"{main_path}/clustering/clusters"
CLUSTERS_FILE  = os.path.join(MO_DATA_DIR,   f"{clustering_name}.h5")
P_FILE         = os.path.join(MO_DATA_DIR,   f"{clustering_name}_P.npz")

P = sp.load_npz(P_FILE)
clusters = load_clusters(CLUSTERS_FILE, P)
clusters = flatten_clusters(clusters)

normal, precursor, extreme = get_num_clusters_each(clusters)
n_clusters_total = len(normal) + len(precursor) + len(extreme)

print(f"Number of clusters ==> Normal: {len(normal)}, Precursor: {len(precursor)}, Extreme: {len(extreme)}.")
print(f"Total Number of clusters: {n_clusters_total}")

all_cluster_centroids = np.array([cluster.center for cluster in normal+precursor+extreme])
all_cluster_centroids = torch.tensor(all_cluster_centroids, device=device, dtype=torch.float32)
cluster_types = torch.tensor([0]*len(normal) + [1]*len(precursor) + [2]*len(extreme), device=device, dtype=torch.int)

Ji = compute_Ji(normal, precursor, extreme)     #inter-cluster variance
Ji = torch.tensor(Ji, device=device, dtype=torch.float32)

# =================================
# Setup solver
nk = 64
re = 40.0
ks = KolSol(nk=nk, nf=4, re=re, ndim=2, device=device)
dt = 0.0025
res = nk * 2

# re40_nk64_train10000s_test10000s_rk4_dt0.0025_Dt0.1_sigma0.6
filename = f"./datasets/fourier_128x128_IC_control.h5"

with h5py.File(filename, "r") as f:
    u_hat = f[f"velocity_train"][:]          

# Copy to avoid overwriting
u_hat0 = torch.tensor(u_hat).detach().clone().to(device=device, dtype=torch.complex64)
u_hat = u_hat0.detach().clone()

# =================================


num_ext_events = 0
num_precursors = 0

control_scale = 4

control_int = 1 * control_scale     # control applied once every "control_int" timesteps (controller lag)
response_time = 1                   # actuator response time in timesteps (actuator lag)
progress_every = 1000               # steps between progress-bar updates; each update makes the CPU wait for the GPU
use_cuda_graph = True               # replay a recording of each control interval (GPU only, without controller or actuator lag)


bk = bk.to(device=device).requires_grad_(False)

from datetime import datetime
run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"Run_{run_suffix}_T{Topt}s_controlInt{control_int}_tR{response_time}"


print(run_name)
print(bk)

Nt = int(Topt / dt) # number of integration timesteps


output_folder = f'{main_path}/output/evaluation/{run_name}'
os.makedirs(output_folder, exist_ok=True)

def beta_from_u_hat(bk, u_hat, ext_threshold, max_beta, encoder, centroids_all, cluster_types, centroids_not_normal, Ji, ks, step, beta, control_int=1):
    u_field = ks.fourier_to_phys(u_hat, nref=res).to(dtype=torch.float32)
    u_field = torch.permute(u_field, (2, 0, 1)).unsqueeze(0)
    u_field = torch.nn.functional.interpolate(u_field, size=output_size, mode='bicubic')

    # latent space representation
    if autoencoder_type == "sae":
        z, _, _, _ = encoder.encoder(u_field, reconstruction_errors=False)  # only the latent vector is used
    else:
        z = encoder.encoder(u_field)  # latent space representation
    z = z.squeeze()
    eps = ks.dissip(u_hat).to(dtype=torch.float32)
    z = torch.cat((z, eps.unsqueeze(0)))
    z = (z - prev_mean ) / prev_std

    z_type = identify_point_type(centroids_all, cluster_types, z, eps, ext_threshold, torch_dist=True, correct_fpee=True)

    if step % control_int == 0:
        diff = z.flatten()[None, :] - centroids_not_normal  
        dist2 = torch.sum(diff**2, axis=1)                  
        e_term = torch.exp(-dist2 / Ji)                     
        num = torch.dot(bk, e_term)
        
        diff_den = z.flatten()[None, :] - centroids_all  
        dist2_den = torch.sum(diff_den**2, dim=-1)       
        e_term_den = torch.exp(-dist2_den / Ji)          
        den = torch.sum(e_term_den, dim=-1)                                 

        beta = num / den

    beta = torch.clip(beta, 0, max_beta)

    return beta, eps, z_type


response_time *= control_scale
actuator = Actuator(response_time, device=device)

ext_threshold = mean_epsilon + nr_dev * std_epsilon

beta = 0


# One RK4 step, updating u_hat in place: the last lines of the old loop, unchanged
def rk4_step_(u_hat, beta):
    k1 = ks.dynamics_control_betaSinVar(u_hat, beta)
    beta1 = beta2 = beta3 = beta

    u_hat_1 = u_hat + 0.5 * dt * k1
    k2 = ks.dynamics_control_betaSinVar(u_hat_1, beta1)

    u_hat_2 = u_hat + 0.5 * dt * k2
    k3 = ks.dynamics_control_betaSinVar(u_hat_2, beta2)

    u_hat_3 = u_hat + dt * k3
    k4 = ks.dynamics_control_betaSinVar(u_hat_3, beta3)

    u_hat += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# One control interval without controller or actuator lag: the controller, then control_scale RK4 steps
def control_interval(u_hat, beta, step):
    beta, eps, z_type = beta_from_u_hat(bk, u_hat, ext_threshold, max_beta, autoencoder, all_cluster_centroids, cluster_types, all_cluster_centroids, Ji, ks, step, beta, control_int=control_int)
    for _ in range(control_scale):
        rk4_step_(u_hat, beta)
    return beta, eps, z_type


# Record one control interval as a CUDA graph: replaying it re-runs the same GPU jobs without the CPU preparing each one
def record_control_interval(u_buf):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up runs on a copy (the interval updates the state in place), so the FFT plans exist
        u_warm = u_buf.clone()
        for _ in range(3):
            control_interval(u_warm, 0, 0)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = control_interval(u_buf, 0, 0)  # beta is recomputed in every interval, so the 0 passed in is not used
    return graph, outputs


# A recording repeats the same GPU jobs every time, so lagged settings (Python logic on beta) run step by step instead,
# as do runs where bk or the state carries a derivative (requires_grad or a forward-mode tangent), so AD is never lost
use_graph = (use_cuda_graph and device == 'cuda' and Nt % control_scale == 0
             and control_int == control_scale and response_time <= control_scale
             and not any(v.requires_grad or fwAD.unpack_dual(v).tangent is not None for v in (bk, u_hat)))
report_every = max(1, progress_every // control_scale)

# One slot per control step, written on the GPU during the run and read back once after it.
# Made once here, so a long run does not keep millions of separate small GPU tensors.
n_measurements = (Nt + control_scale - 1) // control_scale  # Nt / control_scale, rounded up
eps_vals = torch.empty(n_measurements, dtype=torch.float32, device=device)
states_type = torch.empty(n_measurements, dtype=cluster_types.dtype, device=device)
betas = torch.empty(n_measurements, dtype=bk.dtype, device=device)

with torch.no_grad():
    if use_graph:
        graph, (beta, eps, z_type) = record_control_interval(u_hat)
        progress_bar = trange(Nt // control_scale, unit_scale=control_scale)
        for i in progress_bar:
            graph.replay()
            eps_vals[i] = eps  # copies on the GPU into slot i: nothing waits
            states_type[i] = z_type
            betas[i] = beta
            if i % report_every == 0:  # .item() makes the CPU wait for the GPU, so only now and then
                progress_bar.set_postfix({"eps": f"{eps.item():.4f}", "beta": f"{beta.item():.4f}"})
    else:
        progress_bar = trange(Nt)
        for step in progress_bar:
            if step % control_scale == 0:
                slot = step // control_scale
                beta, eps, z_type = beta_from_u_hat(bk, u_hat, ext_threshold, max_beta, autoencoder, all_cluster_centroids, cluster_types, all_cluster_centroids, Ji, ks, step, beta, control_int=control_int)

                eps_vals[slot] = eps
                states_type[slot] = z_type

                if response_time > control_scale:  # prev_beta and new_beta only feed the actuator
                    if step % control_int == 0:
                        if step == 0:
                            new_beta = beta

                        if not (beta == new_beta):
                            prev_beta = new_beta
                            new_beta = beta

                        if step == 0:
                            prev_beta = torch.tensor([1e-6], device=device)

                    beta, prev_beta, new_beta = actuator.get_beta(step+1, prev_beta, new_beta)

                betas[slot] = beta
                if step % progress_every == 0:  # .item() makes the CPU wait for the GPU, so only now and then
                    progress_bar.set_postfix({"eps": f"{eps.item():.4f}", "beta": f"{beta.item():.4f}"})

            rk4_step_(u_hat, beta)

# read back once, after the run, and saved as before: float64 values and default integers
eps_vals = eps_vals.double().cpu().numpy()
states_type = states_type.cpu().numpy().astype(int)
betas = betas.double().cpu().numpy()
num_ext_events = int((states_type == 2).sum())

logging.info(f"Extreme Events: {num_ext_events/Nt*100} %")


np.save(f'{output_folder}/betas.npy', betas)
np.save(f'{output_folder}/eps_vals.npy', eps_vals)
np.save(f'{output_folder}/states_type.npy', states_type)


logging.info(f"Finished the run: {run_name}. Data saved to {output_folder}")