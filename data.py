import mne
import numpy as np
import antropy as ant
from scipy.signal import  hilbert
import pickle
from scipy import stats 
import networkx as nx
import h5py

def main():



    with open("all_epochs.pkl", "rb") as f:
        subjectwise_epochs = pickle.load(f)




    n_subjects = len(subjectwise_epochs)
    n_channels = subjectwise_epochs[0].info['nchan']
    waves = ["delta", "theta", "alpha", "beta", "low_gamma"]
    n_bands = len(waves)
    no_epochs=subjectwise_epochs[0].get_data().shape[0]
    
    # sampen and psd and plv 
    subjectwise_avg_sampen = np.zeros((n_subjects, n_channels, n_bands))
    subjectwise_avg_psd_log = np.zeros((n_subjects, n_channels, 181))
    subjectwise_avg_plv_z = np.zeros((n_subjects, n_bands, n_channels, n_channels))
    subjectwise_per_epoch_sampen=np.zeros((len(subjectwise_epochs),no_epochs,n_channels ,n_bands))
    subjectwise_per_epoch_plv=np.zeros((len(subjectwise_epochs),no_epochs,n_bands,n_channels,n_channels))
    subjectwise_per_epoch_psd=np.zeros((len(subjectwise_epochs),no_epochs,n_channels ,181))

    # ge , cc, cpl,sm
    subjectwise_per_epoch_global_effiy=np.zeros((len(subjectwise_epochs),no_epochs,5))
    subjectwise_per_epoch_cluster_coef=np.zeros_like(subjectwise_per_epoch_global_effiy)
    subjectwise_per_epoch_character_path_len=np.zeros_like(subjectwise_per_epoch_cluster_coef)
    subjectwise_per_epoch_small_worldness=np.zeros_like(subjectwise_per_epoch_cluster_coef)
    subjectwise_avg_ge = np.zeros((len(subjectwise_epochs),n_bands))
    subjectwise_avg_cc = np.zeros_like(subjectwise_avg_ge)
    subjectwise_avg_cpl = np.zeros_like(subjectwise_avg_ge)
    subjectwise_avg_sm =  np.zeros_like(subjectwise_avg_ge)




    for i,sub in enumerate(subjectwise_epochs):
        sampen,psd,plv=metrices(sub)
        global_effiy, cluster_coef, Character_PathLen, small_worldness=network_metric(plv)

        # per epoch allocation 
        subjectwise_per_epoch_sampen[i]=sampen
        subjectwise_per_epoch_plv[i]=plv
        subjectwise_per_epoch_psd[i]=psd

        # allocation to respective array
        sampen_wins = np.empty_like(sampen)
        for ch in range(sampen.shape[1]):
            for wave in range(sampen.shape[2]):
                sampen_wins[:, ch, wave] =stats.mstats.winsorize(sampen[:, ch, wave], limits=[0.05, 0.05])        # Winsorize 5% extremes along subjects axis
            
            subjectwise_avg_sampen[i] = sampen_wins.mean(axis=0)
            subjectwise_avg_psd_log[i] = 10 * np.log10(psd.get_data().mean(axis=0))
            subjectwise_avg_plv_z[i] = 0.5 * np.log((1 + plv.mean(axis=0)) / (1 - plv.mean(axis=0)))      #Fisher z-transform 

            subjectwise_avg_ge[i],subjectwise_avg_cc[i],subjectwise_avg_cpl[i],subjectwise_avg_sm[i] = network_metric(subjectwise_avg_plv_z[i])


            # allocation of the network array
            subjectwise_per_epoch_global_effiy[i]=global_effiy
            subjectwise_per_epoch_cluster_coef[i]=cluster_coef
            subjectwise_per_epoch_character_path_len[i]=Character_PathLen
            subjectwise_per_epoch_small_worldness[i]=small_worldness

        
    #----------------------------#
    """ MAKING FILE """
    #----------------------------#
    with h5py.File("data.h5", "a") as hf:
        hf.create_dataset("sampen", data=subjectwise_avg_sampen)
        hf.create_dataset("plv",data=subjectwise_avg_plv_z)
        hf.create_dataset("psd",data=subjectwise_avg_psd_log)
        hf.create_dataset("sampen_epoch", data=subjectwise_per_epoch_sampen)
        hf.create_dataset("plv_epoch", data=subjectwise_per_epoch_plv)
        hf.create_dataset("psd_epoch",data=subjectwise_per_epoch_psd)
        hf.create_dataset("ge",data=subjectwise_avg_ge)
        hf.create_dataset("cc",data=subjectwise_avg_cc)
        hf.create_dataset("cpl",data=subjectwise_avg_cpl)
        hf.create_dataset("sm",data=subjectwise_avg_sm)
        hf.create_dataset("ge_epoch", data=subjectwise_per_epoch_global_effiy)
        hf.create_dataset("cc_epoch",data=subjectwise_per_epoch_cluster_coef)
        hf.create_dataset("cpl_epoch",data=subjectwise_per_epoch_character_path_len)
        hf.create_dataset("sm_epoch",data=subjectwise_per_epoch_small_worldness)
   






def metrices(sub:mne.Epochs)->np.array: 
    bands = { "delta": (0.5, 4), "theta": (4.0, 7.5), "alpha": (8.0, 12.0),"beta": (12.0, 30.0),"low_gamma": (30.0, 45.0)}
    psd=sub.compute_psd(method='multitaper', fmin=0, fmax=45, picks="eeg", proj=False, remove_dc=True, exclude=(), n_jobs=1, verbose=None)

    freq = sub.info['sfreq']    # sampling frequency

    n_channels=len(sub.ch_names)
    sampen = np.zeros((len(sub), len(sub.ch_names),len(bands) ))
    plv_epochs = []


    for epoch_idx, epoch in enumerate(sub):
        ch_waves = [[mne.filter.filter_data(ch, freq, l_freq=l, h_freq=h)for l, h in bands.values()]for ch in epoch]

        # SampEn for each channel × wave
        for ch_idx, waves_data in enumerate(ch_waves):
            sampen_values = []
            for w in waves_data:
                r = 0.2 * np.std(w)
                se = ant.sample_entropy(w,order=2,tolerance=r) if np.std(w) > 0 else np.nan
                sampen_values.append(se)
            sampen[epoch_idx, ch_idx, :] = np.array(sampen_values)



        #PLV per wave
        plv_epoch = []
        for wave_idx in range(len(bands)):
            #Phases for all channels in this wave
            phases = [np.angle(hilbert(ch_waves[ch_idx][wave_idx])) for ch_idx in range(n_channels)]
            plv_matrix = np.zeros((n_channels,n_channels))
            # PLV between all channel pairs
            for i in range(n_channels):
                for j in range(i+1,n_channels):
                    phase_diff = phases[i] - phases[j]
                    plv_val = np.abs(np.sum(np.exp(1j * phase_diff)) / len(phase_diff))
                    plv_matrix[i, j] = plv_val
                    plv_matrix[j, i] = plv_val
            plv_epoch.append(plv_matrix)
        plv_epochs.append(plv_epoch)

    plv = np.array(plv_epochs)    # shape: (n_epochs, n_waves, n_channels, n_channels)

    return(sampen,psd,plv)


#---------------------
"""helper function"""
#---------------------

def weighted_global_efficiency(G):
    n = len(G)
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    sum_inv = 0.0
    for i in lengths:
        for j, d in lengths[i].items():
            if i != j and d > 0:
                sum_inv += 1.0 / d
    return sum_inv / (n * (n - 1))


def network_metric(sub: np.array):

    eps = 1e-6
    n_random = 10


    def safe_small_world(C_obs, L_obs, C_rand_list, L_rand_list):
        C_rand = max(np.mean(C_rand_list), 1e-8)
        L_rand = max(np.mean(L_rand_list), 1e-8)
        sw = (C_obs / C_rand) / (L_obs / L_rand)
        return np.clip(sw, 0, None)
    

    if sub.ndim == 4:

        no_epoch, waves, nch, _ = sub.shape
        global_effiy = np.zeros((no_epoch, waves))
        cluster_coef = np.zeros_like(global_effiy)
        Character_PathLen = np.zeros_like(global_effiy)
        small_worldness = np.zeros_like(global_effiy)

        for edx, epoch in enumerate(sub):
            for wdx, wave in enumerate(epoch):
                # GE
                G = nx.from_numpy_array(1 / (wave + eps))  # distances

                efficiency = weighted_global_efficiency(G)  # <- make sure this is defined!
                global_effiy[edx, wdx] = efficiency

                # CC
                C = nx.from_numpy_array(wave)
                node_cc = nx.clustering(C, weight="weight")
                cluster_coef[edx, wdx] = np.mean(list(node_cc.values()))

                # CPL
                Character_PathLen[edx, wdx] = nx.average_shortest_path_length(G, weight="weight")

        for edx, epoch in enumerate(sub):
            for wdx, wave in enumerate(epoch):
                C_rand_list, L_rand_list = [], []

                for _ in range(n_random):
                    # create a connected random graph
                    while True:
                        G_rand = nx.gnm_random_graph(nch, G.number_of_edges())
                        if nx.is_connected(G_rand):
                            break

                    # assign properly scaled weights
                    weights = np.random.choice(wave.flatten(), size=G_rand.number_of_edges(), replace=True)
                    weights = weights * (np.mean(wave) / np.mean(weights))
                    for idx, (u, v) in enumerate(G_rand.edges()):
                        G_rand[u][v]["weight"] = weights[idx]

                    # compute clustering coefficient
                    node_cc_rand = nx.clustering(G_rand, weight="weight")
                    C_rand_list.append(np.mean(list(node_cc_rand.values())))

                    # compute average shortest path using distances
                    G_rand_dist = nx.Graph()
                    for u, v in G_rand.edges():
                        G_rand_dist.add_edge(u, v, weight=1 / G_rand[u][v]["weight"])  # invert weight for distance

                    L_rand_list.append(nx.average_shortest_path_length(G_rand_dist, weight="weight"))

                # compute small-worldness
                small_worldness[edx, wdx] = safe_small_world(
                    cluster_coef[edx, wdx],
                    Character_PathLen[edx, wdx],
                    C_rand_list,
                    L_rand_list,
                )
        
        return (global_effiy, cluster_coef, Character_PathLen, small_worldness)



    elif sub.ndim == 3:

        waves, nch, _ = sub.shape
        global_effiy = np.zeros((waves))
        cluster_coef = np.zeros_like(global_effiy)
        Character_PathLen = np.zeros_like(global_effiy)
        small_worldness = np.zeros_like(global_effiy)

        for wdx, wave in enumerate(sub):
            # Global efficiency
            dist_matrix = 1 / (wave + eps)
            G = nx.from_numpy_array(dist_matrix)
            efficiency = weighted_global_efficiency(G)
            global_effiy[wdx] = efficiency

            # Clustering coefficient
            C = nx.from_numpy_array(wave)
            node_cc = nx.clustering(C, weight="weight")
            cluster_coef[wdx] = np.mean(list(node_cc.values()))

            # Characteristic path length
            Character_PathLen[wdx] = nx.average_shortest_path_length(G, weight="weight")

        # Small-worldness
        for wdx, wave in enumerate(sub):
            C_rand_list, L_rand_list = [], []

            for _ in range(n_random):
                # create a connected random graph
                while True:
                    G_rand = nx.gnm_random_graph(nch, G.number_of_edges())
                    if nx.is_connected(G_rand):
                        break

                # assign normalized weights
                weights = np.random.choice(wave.flatten(), size=G_rand.number_of_edges(), replace=True)
                weights = weights * (np.mean(wave) / np.mean(weights))
                for idx, (u, v) in enumerate(G_rand.edges()):
                    G_rand[u][v]["weight"] = weights[idx]

                # compute clustering coefficient
                node_cc_rand = nx.clustering(G_rand, weight="weight")
                C_rand_list.append(np.mean(list(node_cc_rand.values())))

                # compute average shortest path using distances
                G_rand_dist = nx.Graph()
                for u, v in G_rand.edges():
                    G_rand_dist.add_edge(u, v, weight=1 / G_rand[u][v]["weight"])

                L_rand_list.append(nx.average_shortest_path_length(G_rand_dist, weight="weight"))

            # compute small-worldness
            small_worldness[wdx] = safe_small_world(
                cluster_coef[wdx],
                Character_PathLen[wdx],
                C_rand_list,
                L_rand_list,
            )

        return (global_effiy, cluster_coef, Character_PathLen, small_worldness)


    else:
        raise ValueError("Input array must be 3D (waves x channels x channels) or 4D (epochs x waves x channels x channels)")




if __name__ == "__main__":
    main()

