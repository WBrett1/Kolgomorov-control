import numpy as np
import torch
import h5py

class Actuator:
    def __init__(self, response_time, beta_func='cosine', device="cuda"):
        self.response_time = response_time

        self.a = None
        self.b = None
        
        self.current_time = None
        self.start_time = None
        self.expected_end_time = None

        self.new_beta = None
        self.prev_beta = None
        self.beta = None

        self.pi = torch.tensor([np.pi], device=device, dtype=torch.float32)

        self.beta_func = self.cosine_beta if beta_func=='cosine' else self.exponential_beta

    def setup_beta(self, time, beta_1, beta_2):
        self.current_time = time

        # Initializations
        if self.prev_beta is None:
            self.prev_beta = beta_1
        if self.new_beta is None:
            self.new_beta = beta_2
        if self.start_time is None:
            self.start_time = time
        if self.expected_end_time is None:
            self.expected_end_time = time + self.response_time

        # if no input change required
        if beta_1 == beta_2:
            self.new_beta = beta_2
            self.prev_beta = beta_1
            self.start_time = time
            self.expected_end_time = time

        else:
            # if the a new value for beta is given
            if not (beta_2 == self.new_beta):
                # still moving towards the prev value, interrupt and move to the new value
                if time < self.expected_end_time:
                    beta_1 = self.beta
                    self.new_beta = beta_2
                    self.prev_beta = beta_1

                    self.start_time = time
                    self.expected_end_time = time + self.response_time

                # already reached the prev value, use the current betas to move to the new value 
                if time >= self.expected_end_time:
                    # make sure the prev beta is the starting beta value
                    self.new_beta = beta_2
                    self.prev_beta = beta_1
                    self.start_time = time
                    self.expected_end_time = time + self.response_time
            
            # no change in the beta value 
            else:
                # did we reach the end of the update to the target value?
                # if yes, then make sure both betas are the same
                if time >= self.expected_end_time:
                    beta_1 = beta_2
                    self.start_time = time
                    self.expected_end_time = time
                # not yet, then keep the betas and do nothing                    
            
        return beta_1, beta_2

    def exponential_beta(self, time, beta_1, beta_2):
        a = beta_1
        b = (beta_2/beta_1) ** (1/(self.response_time))
        
        if beta_1 < beta_2:
            beta = a * b ** ((time-self.start_time) % self.response_time)
        elif beta_1 > beta_2:
            beta = -1 * a * b ** (- ((time-self.start_time) % self.response_time) + self.response_time) + beta_1
        else:
            beta = beta_2

        return beta
    
    def cosine_beta(self, time, beta_1, beta_2):

        if beta_1 == beta_2:
            beta = beta_2
        else:
            beta = beta_2 + 0.5*(beta_1 - beta_2)*(1 + torch.cos( self.pi * ((time-self.start_time) % self.response_time) / self.response_time ))

        return beta
    
    def get_beta(self, time, beta_1, beta_2):
        beta_1, beta_2 = self.setup_beta(time, beta_1, beta_2)
        beta = self.beta_func(time, beta_1, beta_2)    
        self.beta = beta
        return beta, beta_1, beta_2



# ===========
# CLUSTERING
# ===========

class cluster(object):
    '''Object cluster, defined by it's number (id), the nodes that belong to it, it's center, the clusters to and from which
        it transitions'''
    def __init__(self, nr, nodes, center_coord,center_coord_tess,avg_time, nr_instances, P, extr_clusters, from_clusters):
        self.nr = nr    # cluster index (staring from 0)
        self.nodes = nodes  # node/ hypercube - ids in tessellated space

        is_extreme = 0  # normal state cluster
        if nr in extr_clusters:
            is_extreme=2    # extreme cluster
        elif nr in from_clusters:
            is_extreme=1  # precursor cluster

        self.is_extreme = is_extreme

        clusters_to = P.row[P.col==nr]   # clusters to which the current cluster can transition
        clusters_from = P.col[P.row==nr]     # clusters which can transition to current cluster

        self.transition_to = clusters_to
        self.transition_from = clusters_from

        # cluster center in phase space and tessellated phase space
        self.center = center_coord
        self.center_tess = center_coord_tess

        # average time spent in cluster
        self.avg_time = avg_time

        # number of times the cluster appears in given time series
        self.nr_instances = nr_instances


def load_clusters(fln,P_old):
    hf = h5py.File(fln,'r')
    Nclusters = np.array(hf.get('Nclusters'))
    clusters = []
    extr_clusters = []
    prec_clusters = []
    for i in range(Nclusters):
        nr = np.array(hf.get('C' + str(i) + '_nr'))
        nodes = np.array(hf.get('C' + str(i) + '_nodes'))
        is_extreme = np.array(hf.get('C' + str(i) + '_is_extreme'))
        center_coord = np.array(hf.get('C' + str(i) + '_center'))
        center_coord_tess = np.array(hf.get('C' + str(i) + '_center_tess'))
        avg_time = np.array(hf.get('C' + str(i) + '_avg_time'))
        nr_instances = np.array(hf.get('C' + str(i) + '_nr_instances'))
        if is_extreme==2:
            clusters.append(cluster(nr, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P_old, [nr], [nr-1])) # create a list with the cluster number (to auto-assign is_extreme=2)
            extr_clusters.append(nr)
        elif is_extreme==1:
            clusters.append(cluster(nr, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P_old, [nr-1], [nr])) # create a list with the cluster number (to auto-assign is_extreme=1)
            prec_clusters.append(nr)
        else:
            clusters.append(cluster(nr, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P_old, [nr-1], [nr-1])) # create a list with the cluster number (to auto-assign is_extreme=0)

    hf.close()

    return clusters, extr_clusters, prec_clusters


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

def get_num_clusters_each(clusters):
    normal, precursor, extreme = [], [], []

    for cluster in clusters:
        if cluster.is_extreme == 0:
            normal.append(cluster)
        elif cluster.is_extreme == 1:
            precursor.append(cluster)
        elif cluster.is_extreme == 2: 
            extreme.append(cluster)
    
    assert (len(normal)+len(precursor)+len(extreme)) == len(clusters), "failed to identify all cluster types"

    return normal, precursor, extreme

def compute_Ji(normal, precursor, extreme):
    all_cluster_centroids = np.array([cluster.center     for cluster in normal+precursor+extreme])
    nodes_in_each_cluster = np.array([len(cluster.nodes) for cluster in normal+precursor+extreme])
    nodes_in_each_cluster = np.expand_dims(nodes_in_each_cluster, axis=1) 
    total_nodes_in_trajectory = np.sum(nodes_in_each_cluster)

    trajectory_centroid = np.sum(all_cluster_centroids * nodes_in_each_cluster) / total_nodes_in_trajectory

    Ji = nodes_in_each_cluster * (all_cluster_centroids - trajectory_centroid)**2
    Ji = np.sum(Ji) / total_nodes_in_trajectory

    return Ji


# ===================
def identify_point_type_parallel(centroids_all, cluster_types, state_encoded_sae, dissipation, ext_threshold):
    bsz = dissipation.shape[0]
    num_all_clusters = centroids_all.shape[0]
    
    distance_from_all_clusters = torch.linalg.norm(state_encoded_sae.unsqueeze(1).repeat(1, num_all_clusters, 1) - centroids_all.repeat(bsz, 1, 1), dim=-1)
    
    min_idx = torch.argmin(distance_from_all_clusters, dim=-1)
    point_type = cluster_types[min_idx]

    # correct for extreme event false negatives 
    point_type = torch.where(dissipation >= ext_threshold, 2, point_type)

    return point_type


def identify_point_type(centroids_all, cluster_types, state_encoded_sae, dissipation, ext_threshold, torch_dist=False, correct_fpee=False):
    # decided on the GPU throughout: a Python "if" on dissipation, or indexing with a mask, makes the CPU wait for the GPU
    if torch_dist:
        distance_from_all_clusters = torch.linalg.norm(state_encoded_sae.repeat(centroids_all.shape[0], 1) - centroids_all, dim=1)
    
    if correct_fpee:
        distance_from_all_clusters = distance_from_all_clusters + 1e10 * (cluster_types == 2) # arbitrary large number to avoid selecting z_type = 2
    
    min_idx = torch.argmin(distance_from_all_clusters)
    point_type = cluster_types[min_idx.unsqueeze(0)].squeeze(0)  # indexing with a single-number tensor reads it back to the CPU
    
    return torch.where(dissipation >= ext_threshold, 2, point_type)


def identify_point_type_notAuto(centroids_all, cluster_types, state_encoded_sae, torch_dist=False, correct_fp_ee=False, return_k_closest=False, k_closest=1):

    distance_from_all_clusters = torch.linalg.norm(state_encoded_sae.repeat(centroids_all.shape[0], 1) - centroids_all, dim=1)
    
    if correct_fp_ee:
        distance_from_all_clusters[cluster_types == 2] += 1e10 # arbitrary large number to avoid selecting z_type = 2
    
    min_idx = torch.argmin(distance_from_all_clusters)
    point_type = cluster_types[min_idx]

    if return_k_closest:
        # Get the k smallest distances and their indices
        _, indices = torch.topk(distance_from_all_clusters, k_closest, largest=False)
        
        return point_type, cluster_types[indices]
    else:
        return point_type
