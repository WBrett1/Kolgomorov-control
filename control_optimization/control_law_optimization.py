import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.utils.checkpoint import checkpoint
import torch.autograd.forward_ad as fwAD
import logging
import numpy as np
from tqdm import trange
import scipy.sparse as sp

from parallel_dataloader import ExtEvtsInitConds

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from autoencoders.symmetry_aware_autoencoder import DiscreteSymmetries, SymAutoencoder, STN
from autoencoders.autoencoder import Autoencoder
from KolSol.kolsol.torch.solver import KolSol
from utils import flatten_clusters, get_num_clusters_each, compute_Ji, identify_point_type_parallel, load_clusters


main_path = os.getcwd()
requires_grad = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(999)
np.random.seed(999)

log_to_file = not sys.stdout.isatty()  # True if stdout is not a terminal

# Optimization parameters ========================
lr = 1e-2                   # learning rate
dissip_coeff = 1            # coefficient for energy dissipation rate in the loss function
beta_coeff = 10             # coefficient for beta control signal in the loss function
optimizer_name = "Adam"       # "SGD" or "Adam" or "AdamW"
init_cond = "normal"            # "zeros" or "normal"
autoencoder_type = "sae"        # "sae": symmetry-aware autoencoder or "cae": autoencoder

bsz = 5                         # batch size
accumulation_steps = 1          # gradient accumulation steps

max_training_sims = 5
validation_freq = 5             # validate every n epochs

Topt = 30.8                  # Temporal horizon, i.e., length of each simulation 

mean_epsilon, std_epsilon = 0.11726184557297091, 0.03806973962448417  # energy dissipation rate mean and std from entire dataset

# latent space normalization params based on the dataset used for clustering
if autoencoder_type == "sae":
    prev_mean = torch.tensor([[  3.0741701126098633, -2.3254714012145996, 3.2388222217559814, -0.9291426539421082, -36.08496856689453, 3.460550308227539, 6.827239990234375, 18.972339630126953, 0.11595899091001875]], device=device, dtype=torch.float32) 
    prev_std = torch.tensor([[ 6.697792053222656, 6.32340145111084, 5.581307888031006, 7.944993019104004, 6.816481590270996, 5.449934482574463, 7.902254104614258, 6.43773078918457, 0.0375438561795101 ]], device=device, dtype=torch.float32)
else:
    prev_mean = torch.tensor([[  -1.3589740991592407, -0.885854959487915, 3.369107723236084, -2.049339771270752, -3.06653094291687, 1.381832480430603, 5.244030952453613, -1.4049558639526367, 0.11595899091001875]], device=device, dtype=torch.float32) 
    prev_std = torch.tensor([[ 25.471277236938477, 7.641894817352295, 18.170530319213867, 10.358161926269531, 7.718150615692139, 26.935338973999023, 22.47753143310547, 11.527128219604492, 0.0375438561795101 ]], device=device, dtype=torch.float32)

nr_dev = 1.5                # number of std deviations above the mean to consider an extreme event

output_size = (32, 32)      # interpolation output, used as input to the autoencoder

max_beta = 0.8              # max value for the control signal
# END Optimization parameters ========================


def beta_from_u_hat(bk, u_hat, ext_threshold, max_beta, encoder, centroids_all, cluster_types, centroids_not_normal, Ji, ks, step, control_int=1):
    batch_size = u_hat.shape[0]
    n_vars = bk.shape[-1]

    u_field = torch.stack([ks.fourier_to_phys(u_hat[b], nref=res).to(dtype=torch.float32) for b in range(batch_size)], dim=0)
    u_field = torch.permute(u_field, (0, 3, 1, 2))
    u_field = torch.nn.functional.interpolate(u_field, size=output_size, mode='bicubic')

    if autoencoder_type == "sae":
        z, _, _, _ = encoder.encoder(u_field, reconstruction_errors=False)  # only the latent vector is used
    else:
        z = encoder.encoder(u_field)

    z = z.squeeze()

    if len(z.shape) == 1 and batch_size == 1:
        z = z.unsqueeze(0)

    eps = [ks.dissip(u_hat[b]).to(dtype=torch.float32) for b in range(batch_size)] 
    eps = torch.stack(eps, dim=0)

    z = torch.cat( (z, eps.unsqueeze(1)),  dim=1)
    z = (z - prev_mean ) / prev_std

    z_type = identify_point_type_parallel(centroids_all, cluster_types, z.clone(), eps.clone(), ext_threshold)

    if step % control_int == 0:
        diff = z.unsqueeze(1).repeat(1, n_vars, 1) - centroids_not_normal.unsqueeze(0).repeat(batch_size, 1, 1) 
        dist2 = torch.sum(diff**2, dim=-1)                         
        e_term = torch.exp(-dist2 / Ji)                            
        num = torch.sum(bk.repeat(batch_size, 1) * e_term, dim=-1) 
        
        diff_den = z.unsqueeze(1).repeat(1, centroids_all.shape[0], 1) - centroids_all.unsqueeze(0).repeat(batch_size, 1, 1) 
        dist2_den = torch.sum(diff_den**2, dim=-1)                     
        e_term_den = torch.exp(-dist2_den / Ji)                        
        den = torch.sum(e_term_den, dim=-1)                            
        
        beta = num / den
    
    beta = torch.clip(beta, 0, max_beta)

    return beta, eps, z_type


def update_u_hat(u_hat, beta):
    """ RK4 time-stepping with control term """
    batch_size = u_hat.shape[0]

    k1 = torch.stack([ks.dynamics_control_betaSinVar(u_hat[b], beta[b]) for b in range(batch_size)], dim=0)
    u_hat_1 = u_hat + 0.5 * dt * k1

    k2 = torch.stack([ks.dynamics_control_betaSinVar(u_hat_1[b], beta[b]) for b in range(batch_size)], dim=0)
    u_hat_2 = u_hat + 0.5 * dt * k2
    
    k3 = torch.stack([ks.dynamics_control_betaSinVar(u_hat_2[b], beta[b]) for b in range(batch_size)], dim=0)
    u_hat_3 = u_hat + dt * k3
    
    k4 = torch.stack([ks.dynamics_control_betaSinVar(u_hat_3[b], beta[b]) for b in range(batch_size)], dim=0)
    
    u_hat = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return u_hat


# Mean of the selected entries, worked out on the GPU (NaN when none are selected)
def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


# The per-step means that exist (not NaN), read back from the GPU in one go
def present(means):
    means = torch.stack(means).double().cpu()  # one copy back from the GPU, so the .item() calls below do not wait
    return [m for m in (v.item() for v in means) if not np.isnan(m)]  # .item(), unlike .numpy(), also works inside torch.func.jvp


def loss_function(bk, u_hat0, encoder, centroids_all, cluster_types, centroids_not_normal, Ji, ks, dt, Topt, 
                  mean_epsilon, std_epsilon, f_num_ext_events=False,
                  scale_epsilon=1.0, scale_beta=1.0, control_int=1):
    
    u_hat = u_hat0

    # checkpoint cannot carry forward-mode (dual) tangents, so forward-mode AD runs each step without it
    forward_mode = any(fwAD.unpack_dual(v).tangent is not None for v in (bk, u_hat0))
    run_step = (lambda fn, *args: fn(*args)) if forward_mode else checkpoint

    Nt = int(Topt / dt)

    eps_total = torch.tensor([0], device=device, dtype=torch.float32)
    beta_penalty_total = torch.tensor([0], device=device, dtype=torch.float32)

    num_ext_events = 0
    num_f_ext_events = 0
    num_precursors = 0
    betas_norm = []
    betas_prec = []
    betas_ext = []
    nan_found = torch.zeros((), dtype=torch.bool, device=device)
    no_type_found = torch.zeros((), dtype=torch.bool, device=device)
    control_int = 1
    ext_threshold = mean_epsilon + nr_dev * std_epsilon

    critical_eps = mean_epsilon + 0.5 * std_epsilon

    all_eps = []

    for step in trange(Nt, disable=log_to_file):
        beta, eps, z_type = run_step(beta_from_u_hat, bk, u_hat, ext_threshold, max_beta, encoder, centroids_all, cluster_types, centroids_not_normal, Ji, ks, step, control_int)
    
        nan_found |= torch.isnan(eps).any()  # checked once after the loop: an assert here made the CPU wait every step
        
        all_eps.append(eps.detach())

        is_extreme = eps >= ext_threshold
        is_critical = eps >= critical_eps
        
        coeff = (is_extreme) * 10 + (is_critical) * 9 +  + 1           # 1 if normal and 10 if Critical and 20 if Extreme
        eps_total += (torch.abs(((eps-mean_epsilon)/mean_epsilon)) * coeff).mean()

        
        num_precursors += (z_type == 1).sum()

        num_ext_events += is_extreme.sum()    # (i.e., identified as an extreme state)
        num_f_ext_events += (eps >= mean_epsilon + 2 * std_epsilon).sum()    # (i.e., identified as actual extreme state)
                

        is_ext = (z_type == 2) | (is_extreme)
        betas_ext.append(masked_mean(beta.detach(), is_ext))

        is_prec = (z_type == 1) & ~is_extreme
        betas_prec.append(masked_mean(beta.detach(), is_prec))

        is_norm = (z_type == 0) & ~is_extreme
        betas_norm.append(masked_mean(beta.detach(), is_norm))

        no_type_found |= (is_ext.sum() + is_prec.sum() + is_norm.sum() == 0)

        is_normal = (z_type == 0)
        coeff_beta = 9 * (is_normal) + 1        # 10 if normal, 1 otherwise
        beta_penalty_total += (torch.abs(beta) * coeff_beta).mean()

        u_hat = run_step(update_u_hat, u_hat, beta)

    # the counts and statistics above stayed on the GPU; read them back once
    assert not nan_found, "Tensor contains NaN values"
    if no_type_found:
        print("Unknown type of cluster caught!")
    num_precursors, num_ext_events, num_f_ext_events = int(num_precursors), int(num_ext_events), int(num_f_ext_events)
    betas_norm, betas_prec, betas_ext = present(betas_norm), present(betas_prec), present(betas_ext)

    eps_total = eps_total * dt
    beta_penalty_total = beta_penalty_total * dt

    loss = 1/Topt*((scale_epsilon * eps_total) + (scale_beta * beta_penalty_total))
    
    all_eps = torch.cat(all_eps).cpu()  # the same values, on the CPU as before
    # TODO: remove unnecessary logging
    logging.info(f"Max Eps: {torch.max(all_eps).item()}, ExtvtsThreshold: {ext_threshold}, Mean Eps: {torch.mean(all_eps).item()}")    

    logging.info(f"DissipLoss: {(eps_total/Topt).item():.5}, Precusors: {num_precursors}, ExtEvents: {num_f_ext_events}")
    logging.info(f"BetaLoss: {(beta_penalty_total/Topt).item():.5}")

    if len(betas_norm) > 0:
        logging.info(f"NORMAL BetaMean: {np.mean(betas_norm):.3}, BetaMax: {np.max(betas_norm):.3}, BetaMin: {np.min(betas_norm):.3}")
    if len(betas_prec) > 0:
        logging.info(f"PRECUR BetaMean: {np.mean(betas_prec):.3}, BetaMax: {np.max(betas_prec):.3}, BetaMin: {np.min(betas_prec):.3}")
    if len(betas_ext) > 0:
        logging.info(f"EXTEVT BetaMean: {np.mean(betas_ext):.3}, BetaMax: {np.max(betas_ext):.3}, BetaMin: {np.min(betas_ext):.3}")
    logging.info(f"Loss: {(loss).item():.5}")
    
    if f_num_ext_events:
        return loss, num_f_ext_events
    else:
        return loss, num_ext_events


# Setup logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# 1.a. Load SAE/CAE autoencoder
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


# 2. Setup solver
ks = KolSol(nk=64, nf=4, re=40, ndim=2, device=device)
dt = 0.0025 * 4
re = 40
res = 64 * 2

# 3. Loading initial conditions for extreme events
datafile = f"./datasets/fourier_128x128_IC_multiple_EEs.h5"


def objective(bk, u_input, f_num_ext_events=False):
    return loss_function(
        bk=bk,
        u_hat0=u_input,
        encoder=autoencoder,
        centroids_all = all_cluster_centroids, 
        cluster_types = cluster_types,
        centroids_not_normal = all_cluster_centroids,
        Ji=Ji,
        ks=ks,
        dt=dt,
        Topt=Topt,
        mean_epsilon=mean_epsilon,
        std_epsilon=std_epsilon,
        scale_epsilon=dissip_coeff,
        scale_beta=beta_coeff,       
        f_num_ext_events=f_num_ext_events
    )


print("Starting optimization process...")

# ===========

# start the optimization from a uniform distribution with min = 0 and max = 0.1 (exclusive)
if "zeros" in init_cond:
    print("Using zeros for initial guess")
    bk = torch.zeros((n_clusters_total,))
elif "normal" in init_cond:
    print("Using normal distribution * 0.1 for initial guess")
    bk = torch.randn((n_clusters_total,)) * 0.1
else:
    raise NotImplementedError("Requested initial condition is not available.")


bk = bk.to(dtype=torch.float32, device=device).requires_grad_(requires_grad)
print("Number of optimization vars:", bk.shape[0])

epochs = 5000       # number of max training iterations

losses = []

optimizer = None
if optimizer_name == "SGD":
    optimizer = torch.optim.SGD([bk], lr=lr)
elif optimizer_name == "Adam":
    optimizer = torch.optim.Adam([bk], lr=lr)
elif optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW([bk], lr=lr)
else:
    raise NotImplementedError("Requested optimizer is not available.")

optimizer.zero_grad()


# validation dataset
dataset_valid = ExtEvtsInitConds(main_path, datafile=datafile, num_sims=20, device=device)
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
num_batches_valid = len(dataloader_valid)


num_sims = 0       # start with 0 simulations (i.e., start by loading a new initial condition for the first iteration)
shuffle = False
num_ext_events = 0

for epoch in range(epochs):
    print("=========== Epoch", epoch+1, "Lr", lr, "===========")

    epoch_loss = 0
    
    if num_ext_events == 0 and num_sims <= max_training_sims - 1:
        num_sims += 1 

        dataset = ExtEvtsInitConds(main_path, datafile=datafile, num_sims=num_sims, device=device)
        print("Number of initial conditions loaded:", len(dataset))
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle)
        num_batches = len(dataloader)

    num_ext_events = 0
    for j, u_hat_input in enumerate(dataloader):
        print("Batch", j+1, "of", num_batches)

        loss, curr_num_ext_events = objective(bk, u_hat_input)
        num_ext_events += curr_num_ext_events

        loss = loss  / accumulation_steps

        epoch_loss += loss.item() * len(u_hat_input) / len(dataset)
        
        loss.backward()
        
        if (j + 1) % accumulation_steps == 0:
            optimizer.step()   
            optimizer.zero_grad()

    print("bk:", bk)
    epoch_loss = epoch_loss * accumulation_steps
    losses.append(epoch_loss)
    logging.info(f"Epoch loss: {epoch_loss}")

    if epoch % validation_freq == 0:
        logging.info(f"Starting validation...")

        with torch.no_grad():
            losses_valid = []

            for j, u_hat_input in enumerate(dataloader_valid):
                if j < max_training_sims - 1:
                    # skip the first training sims
                    continue
                print("Sim", j+1, "of", num_batches_valid)
                loss_val, f_num_ext_events = objective(bk, u_hat_input, f_num_ext_events=True)
                
                losses_valid.append(loss_val.item())

                if f_num_ext_events > 0:
                    logging.info("Extreme event triggered! Stopping validation...")
                    break

            logging.info(f'Mean Validation loss: {np.mean(losses_valid)}')
            logging.info("Validation done!")
    
    if epoch_loss == 0:
        print("Optimization stopped due to 0 loss!")
        break


print("Optimization completed. Final b_k:")
print(bk)