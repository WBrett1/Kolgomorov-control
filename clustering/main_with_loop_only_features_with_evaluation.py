import numpy as np
import matplotlib.pyplot as plt
import clustering_func_only_features as clfunc
from scipy.integrate import simpson
import scipy.sparse as sp
import time
import h5py
import pickle
import os
import sys
from tqdm import trange
import torch
import torch.nn as nn
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autoencoders.symmetry_aware_autoencoder import DiscreteSymmetries, SymAutoencoder, STN
from autoencoders.autoencoder import Autoencoder

from KolSol.kolsol.numpy.solver import KolSol

from control_optimization.utils import get_num_clusters_each, identify_point_type, identify_point_type_notAuto


# =============================================================================
# Read the data
# =============================================================================

main_path = os.getcwd()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(0)
torch.manual_seed(0)

# Clustering parameters ========================
is_sae = True           # use symmetry-aware autoencoder
is_pod = False          # use linear autoencoder
is_vCAE = False         # use vanilla nonlinear autoencoder
M = 30                  # number of hypercubes per dimension
stride = 4              # data stride from the training data
final_time_sec = 2000   # total time in seconds

z_size = 8              # latent space dimension
nr_dev = 1.5            # number of std deviations above the mean to consider an extreme event

min_clusters = 100      # minimum number of clusters to identify
max_it = 4              # maximum number of iterations for clustering

stationary_mean_dissipation, stationary_std_dissipation = 0.11726184557297091, 0.03806973962448417  # energy dissipation rate mean and std from entire dataset

bsz = 100               # batch size for autoencoder encoding
# END Clustering parameters ========================


assert is_sae + is_pod + is_vCAE <= 1, "Can\'t have more than one autoencoder type at a time!"
assert is_sae + is_pod + is_vCAE > 0

fld = f"{main_path}/clustering/clusters/"


nk = 64                     # number of symmetric fourier modes in each dimension
training_split = "test"
test_split = 'test'

re = 40             # Reynolds number

dt = 0.1
final_time = int(final_time_sec / dt)
dt *= stride


if is_sae:
    suffix = f'_sae'
elif is_vCAE:
    suffix = f'_vcae'
else:
    suffix = f'_pod'


data_file = f"{main_path}/datasets/phys_128x128.h5"

#  ================
# Load data 
with h5py.File(data_file, "r") as f:
    velocity_torch = f[f"velocity_{training_split}"][0:final_time:stride]      # (N, 32, 32, 2)
    dissipation = f[f"dissipation_{training_split}"][0:final_time:stride]      # (N,)

print("loaded total seconds:", velocity_torch.shape[0] * dt, "s") 

velocity_torch = torch.tensor(velocity_torch, dtype=torch.float32).squeeze(1).to(device=device)
dissipation = np.array(dissipation)

mean_dissipation = np.mean(dissipation)
std_dissipation = np.std(dissipation)
ee_dissipation = mean_dissipation + nr_dev * std_dissipation


#  ================
# Loading autoencoder

if is_sae:
    state_dict = torch.load(f"{main_path}/autoencoders/pretrained_models/sae_autoencoder_{z_size}.pth", map_location=device)
    stn = STN((2,32,32))
    discrete_symmetries = DiscreteSymmetries().to(device=device)
    autoencoder_base = Autoencoder(latent_dim=z_size, base=32, activation=nn.Tanh()).to(device=device)
    autoencoder = SymAutoencoder(autoencoder_base, stn, discrete_symmetries).to(device=device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

elif is_vCAE:
    state_dict = torch.load(f"{main_path}/autoencoders/pretrained_models/autoencoder_{z_size}.pth", map_location=device)
    autoencoder = Autoencoder(latent_dim=z_size, base=32, activation=nn.Tanh()).to(device=device)

else:
    state_dict = torch.load(f"{main_path}/autoencoders/pretrained_models/linear_autoencoder_{z_size}.pth", map_location=device)
    autoencoder = Autoencoder(latent_dim=z_size, base=32, activation=nn.Identity()).to(device=device)

autoencoder.load_state_dict(state_dict)
autoencoder.eval()

#  ================
# Data encoding
org_frames = velocity_torch.shape[0]
num_batches = org_frames // bsz
U_sae_enc = []

for batch in trange(num_batches):
    start = 0
    end = org_frames // num_batches
    
    if is_sae:
        encoded, _, _, _ = autoencoder.encoder(velocity_torch[start:end])
    else:
        encoded = autoencoder.encoder(velocity_torch[start:end])

    velocity_torch = velocity_torch[org_frames // num_batches:]
    
    encoded = encoded.detach().cpu().squeeze()
    U_sae_enc.append(encoded)

if bsz > 1:
    x = np.concatenate(U_sae_enc, axis=0)
else:
    x = np.array(U_sae_enc)

print("Done with data encoding!")

# normalization
Ndim = x.shape[1] # second dimension is the list of elements


mean_data = np.array(list(np.mean(x,0)) + [mean_dissipation]).reshape((1,Ndim+1))
std_data = np.array(list(np.std(x,0)) + [std_dissipation]).reshape((1,Ndim+1))

print(list(mean_data[0]))
print(list(std_data[0]))

x = (x - np.mean(x,0).reshape((1,Ndim)) ) / np.std(x,0).reshape((1,Ndim))

dissipation_normed = (dissipation - mean_dissipation) / std_dissipation
dissipation_normed = np.expand_dims(dissipation_normed, axis=1)

x = np.concatenate([x, dissipation_normed], axis=1)

Nt = x.shape[0] # first dimension of x is time
t = np.arange(Nt) * dt
tf = Nt * dt

# =============================================================================
# Some preparatory lists and hyperparameters
# =============================================================================

prediction_time_list = []
false_positive_list = []
false_negative_list = []
instances_list = []

# Provide the hyperparameter for the clustering

extr_dim = [x.shape[1]-1] # the dimension to use for extreme events identification is the final one
plotting = False

features = [] # to give names to the features
for i in range(x.shape[1]):
    features.append('x' + str(i))

# =============================================================================
# Some preparatory lists and hyperparameters
# =============================================================================

ee_threshold = stationary_mean_dissipation + nr_dev * stationary_std_dissipation
ee_threshold = (ee_threshold - mean_dissipation ) / std_dissipation

clusters, D, P, P_graph, tess_ind, tess_ind_trans, tess_ind_cluster = clfunc.extreme_event_identification_process(
        t, x, M, extr_dim, 'Example', min_clusters, max_it, nr_dev, features, plotting, 'classic', False, ee_threshold=ee_threshold)


labels = tess_ind_cluster[tess_ind]

def centroid_of_cluster(cid):
    mask = (tess_ind_cluster[tess_ind] == cid)
    return x[mask].mean(axis=0)
C = {cid: centroid_of_cluster(cid) for cid in np.unique(labels)}

# ================================
# Save the identified clusters

fln_r = fld + f'clusters{suffix}'

fln = fln_r + '_D.npz'
sp.save_npz(fln,D)
fln = fln_r + '_P.npz'
sp.save_npz(fln,P)
fln = fln_r + '_P_graph.pickle'
pickle.dump(P_graph,open(fln,'wb'))
fln = fln_r + '_tess_ind_data.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('tess_ind',data=tess_ind)
hf.create_dataset('tess_ind_trans',data=tess_ind_trans)
hf.create_dataset('tess_ind_cluster',data=tess_ind_cluster)
hf.close()

fln = fln_r + '.h5'
clfunc.save_clusters(fln,clusters)


# Check on "new" data series
# Tessellate data set (without extreme event identification)
x_tess, temp = clfunc.tesselate(x, M, extr_dim, nr_dev, ee_threshold=ee_threshold)
x_tess = clfunc.tess_to_lexi(x_tess, M, x.shape[1])

# Translate data set to already identified clusters
x_clusters = clfunc.data_to_clusters(x_tess, D, x, clusters)

is_extreme = np.zeros_like(x_clusters)
for cluster in clusters:
    # New data series, determining whether the current
    # state of the system is extreme (2), precursor (1) or normal state (0)
    is_extreme[np.where(x_clusters == cluster.nr)] = cluster.is_extreme

# np.save(f"{fld}is_extreme{suffix}", np.array(is_extreme))
# np.save(f"{fld}dissipation{suffix}", np.array(dissipation))

# Calculate the false positive and false negative rates
avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme, instances_precursor_after_extreme = clfunc.backwards_avg_time_to_extreme(
        is_extreme, dt)
print('Average time from precursor to extreme:', avg_time, ' seconds')
print('Nr times when extreme event had a precursor:', instances)
print('Nr extreme events without precursors (false negative):',
          instances_extreme_no_precursor)
print('Percentage of false negatives:', instances_extreme_no_precursor /
          (instances+instances_extreme_no_precursor)*100, ' %')
print('Percentage of extreme events with precursor (true positives):',
          instances/(instances+instances_extreme_no_precursor)*100, ' %')
print('Nr precursors without a following extreme event (false positives):',
          instances_precursor_no_extreme)
print('Percentage of false positives:', instances_precursor_no_extreme /
          (instances+instances_precursor_no_extreme)*100, ' %')


prediction_time_list.append(avg_time)
false_negative_list.append(instances_extreme_no_precursor)
false_positive_list.append(instances_precursor_no_extreme)
instances_list.append(instances)


# ======================
#  General postprocessing

print('\n\nSTARTING EVALUATION USING DISTANCE-BASED METHOD...\n\n')

colormap = { 0: "blue", 1: "orange", 2: "red"}


start_idx = 20000
end_idx = 100000
dt = 0.1

with h5py.File(data_file, "r") as f:
    velocity = f[f"velocity_{test_split}"][start_idx:end_idx]            # (N, 32, 32, 2)
    dissipation = f[f"dissipation_{test_split}"][start_idx:end_idx]      # (N,)


velocity = torch.tensor(velocity, dtype=torch.float32).squeeze(1).to(device=device)

org_frames = velocity.shape[0]
num_batches = org_frames // bsz
U_sae_enc = []

for batch in trange(num_batches):
    end = org_frames // num_batches
    
    if is_sae:
        encoded, _, _, _ = autoencoder.encoder(velocity[:end].cuda())
    else:
        encoded = autoencoder.encoder(velocity[:end].cuda())
    velocity = velocity[org_frames // num_batches:]
    
    encoded = encoded.detach().cpu().squeeze()
    U_sae_enc.append(encoded)


if bsz > 1:
    x = np.concatenate(U_sae_enc, axis=0)
else:
    x = np.array(U_sae_enc)
    
print("Done with data encoding!")


Ndim = x.shape[1] # second dimension is the list of elements
prev_mean, prev_std = mean_data, std_data

x = np.concatenate([x, np.expand_dims(dissipation, axis=1)], axis=1)

x = (x - prev_mean ) / prev_std     # standardize dataset


MO_DATA_DIR    = f"{main_path}/clustering/clusters"
CLUSTERS_FILE  = os.path.join(MO_DATA_DIR,   f"clusters{suffix}.h5")
P_FILE         = os.path.join(MO_DATA_DIR,   f"clusters{suffix}_P.npz")
D_FILE         = os.path.join(MO_DATA_DIR,   f"clusters{suffix}_D.npz")
TESS_FILE      = os.path.join(MO_DATA_DIR,   f"clusters{suffix}_tess_ind_data.h5")

def flatten_clusters(raw):
    flat = []
    def visit(item):
        if hasattr(item, "nr"):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                visit(sub)
    visit(raw)
    return flat

P = sp.load_npz(P_FILE)
clusters = clfunc.load_clusters(CLUSTERS_FILE, P)
clusters = flatten_clusters(clusters)
D = sp.load_npz(D_FILE)


x_clusters = []

actual_ee = 0
identified_ee = 0
false_ee = 0

# =======
normal, precursor, extreme = get_num_clusters_each(clusters)
n_clusters_total = len(normal) + len(precursor) + len(extreme)

centroids_all = np.array([cluster.center for cluster in normal+precursor+extreme])
centroids_all = torch.tensor(centroids_all, dtype=torch.float32)
cluster_types = torch.tensor([0]*len(normal) + [1]*len(precursor) + [2]*len(extreme), dtype=torch.int)

ext_threshold = stationary_mean_dissipation + nr_dev * stationary_std_dissipation

is_extreme = []

for state_idx in trange(x.shape[0]):
    state_enc_sae = torch.tensor(x[state_idx])
    
    z_type = identify_point_type_notAuto(centroids_all, cluster_types, state_enc_sae, torch_dist=True)

    if dissipation[state_idx] >= ext_threshold:
        actual_ee += 1

        if z_type == 2:
            identified_ee += 1
    else:
        if z_type == 2:
            false_ee += 1 

    is_extreme.append(z_type.item())


colors = []
is_extreme = np.array(is_extreme)

for point in range(len(is_extreme)):
    colors.append(colormap[is_extreme[point]])


avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme, instances_precursor_after_extreme = clfunc.backwards_avg_time_to_extreme(
        is_extreme, dt)
print('Average time from precursor to extreme:', avg_time, ' seconds')
print('Nr times when extreme event had a precursor:', instances)
print('Nr extreme events without precursors (false negative):',
          instances_extreme_no_precursor)
print('Percentage of false negatives:', instances_extreme_no_precursor /
          (instances+instances_extreme_no_precursor)*100, ' %')
print('Percentage of extreme events with precursor (true positives):',
          instances/(instances+instances_extreme_no_precursor)*100, ' %')
print('Nr precursors without a following extreme event (false positives):',
          instances_precursor_no_extreme)
print('Percentage of false positives:', instances_precursor_no_extreme /
          (instances+instances_precursor_no_extreme)*100, ' %')
print(f'{identified_ee = }, {actual_ee = }, True Identified ExtEvts: {identified_ee / actual_ee * 100}%')
print(f'{false_ee = }, False ExtEvts out of total identified ExtEvts: {false_ee / (false_ee + identified_ee) * 100}%')

ids_norm, ids_prec, ids_ext = [], [], []
cent_norm, cent_prec, cent_ext = [], [], []
for cl in clusters:
    cid = int(np.asarray(cl.nr).item())
    if cl.is_extreme == 0:
        ids_norm.append(cid);  cent_norm.append(cl.center)
    elif cl.is_extreme == 1:
        ids_prec.append(cid);  cent_prec.append(cl.center)
    elif cl.is_extreme == 2:
        ids_ext.append(cid);   cent_ext.append(cl.center)
cent_norm = np.vstack(cent_norm)
cent_prec = np.vstack(cent_prec)
cent_ext  = np.vstack(cent_ext)
print(f"Normal: {len(ids_norm)}, Precursors: {len(ids_prec)}, Extreme: {len(ids_ext)} centroids.", f"\nTotal: {len(ids_norm) + len(ids_prec) + len(ids_ext)}")

