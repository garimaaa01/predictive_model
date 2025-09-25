
import pandas as pd 
import bct
from scipy import stats 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pickle
import sys
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm  
from statsmodels.regression.linear_model import OLS


def main():
    with open("all_epochs.pkl", "rb") as f:
        subjectwise_epochs = pickle.load(f)


    with h5py.File("data.h5","r") as hf:

        ch_list = subjectwise_epochs[0].info["ch_names"]
        info = subjectwise_epochs[0].info
        group , result = covaries(hf)

        if len(sys.argv)==2 and sys.argv[1]=="viz":
            viz_histogram(hf)
            viz_violinplot(group, result , ch_list, info)
           

        return (group,result)
            
      
      
#-----------------------
# VISUALIZATION    
#-----------------------

def viz_histogram(hf):

    band_names = ["delta", "theta", "alpha", "beta", "low_gamma"]
    colors = ["red", "green", "blue", "black", "purple"]

    # HISTOGRAM/ KDE
    #----------------
    sampen = hf["sampen_epoch"][:]
    psd = psd_freq(hf["psd_epoch"][:])
    plv = hf["plv_epoch"][:]
    ge = hf["ge_epoch"][:]
    cc = hf["cc_epoch"][:]
    cpl = hf["cpl_epoch"][:]
    sm = hf["sm_epoch"][:]

    metrics_dict = {
    "sampen": sampen,
    "psd": psd,
    "plv": plv,
    "ge": ge,
    "cc": cc,
    "cpl": cpl,
    "sm": sm
            }


    for m, arr in metrics_dict.items():
        if arr.ndim == 4 :  # 4D: subjects x something x something x waves
            for i, (band, color) in enumerate(zip(band_names, colors)):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                ad_vals = arr[:36, :, :, i].flatten()
                hc_vals = arr[36:, :, :, i].flatten()

                sns.histplot(ad_vals, kde=True, bins=30, color=color, alpha=0.6, label=band, ax=axes[0])
                sns.histplot(hc_vals, kde=True, bins=30, color=color, alpha=0.6, label=band, ax=axes[1])

                axes[0].set_title(f"AD — {m} — Wave {i+1}")
                axes[0].set_xlabel(f"{m}")
                axes[0].set_ylabel("Count")

                axes[1].set_title(f"HC — {m} — Wave {i+1}")
                axes[1].set_xlabel(f"{m}")
                axes[1].set_ylabel("Count")

                axes[0].legend()
                axes[1].legend()
                plt.tight_layout()
                plt.show()

  
        elif arr.ndim == 3:  # 3D: subjects x something x waves
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            for i, (band, color) in enumerate(zip(band_names, colors)):
                ad_vals = arr[:36, :, i].flatten()
                hc_vals = arr[36:, :, i].flatten()

                sns.histplot(ad_vals, kde=True, bins=30, color=color, alpha=0.6, label=band, ax=axes[0])
                sns.histplot(hc_vals, kde=True, bins=30, color=color, alpha=0.6, label=band, ax=axes[1])

            axes[0].set_title(f"AD — {m} — All Waves")
            axes[0].set_xlabel(f"{m}")
            axes[0].set_ylabel("Count")

            axes[1].set_title(f"HC — {m} — All Waves")
            axes[1].set_xlabel(f"{m}")
            axes[1].set_ylabel("Count")

            axes[0].legend()
            axes[1].legend()
            plt.tight_layout()
            plt.show()

def viz_violinplot(groups:dict, result:dict  , ch_list:dict ,info):

    bands = ["delta", "theta", "alpha", "beta", "low_gamma"]
    metric = ["sampen","psd","plv","ge","cc","cpl","sm"]
    grp  = ["AD", "HC"]



    for m in metric:
        if  not m in ["gender", "age"]:
            df = []
            for g in grp:
                arr = groups[g][m]    
                if arr.ndim == 3:
                    sub , ch , wave = np.indices(arr.shape)
                    if arr.shape[1] > arr.shape[2]:
                        df.append( pd.DataFrame({
                            "subject": sub.ravel(),
                            "channel":ch.ravel(),
                            "band":[bands[i] for i in wave.ravel()],
                            "values": arr.ravel(),
                            "metric":m,
                        "group":g,
                        }))
                    else:
                        sub, wave, edge = np.indices(arr.shape)
                        df.append(pd.DataFrame({
                                "subject": sub.ravel(),
                                "band": [bands[i] for i in wave.ravel()],
                                "edge": edge.ravel(),
                                "values": arr.ravel(),
                                "group": g,
                                "metric":m     # or "HC", depending on the subject
                            }))
                        
                elif arr.ndim == 2:
                    sub , wave = np.indices(arr.shape)
                    df.append(pd.DataFrame({
                        "subject": sub.ravel(),
                        "band":[bands[i] for i in wave.ravel()],
                        "values": arr.ravel(),
                        "metric":m,
                        "group":g
                    }))          
            
            plt.figure(figsize=(15,6))
            sns.boxplot(
                    data=pd.concat(df),
                    x="band",
                    y="values",
                    hue="group",
                    dodge=True
                    )
            plt.title(f"{m} — AD vs HC")
            plt.ylabel(f"{m}value")
            plt.xlabel("Frequency band")
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(15,6))
            sns.violinplot(
                    data=pd.concat(df),
                    x="band",
                    y="values",
                    hue="group",
                    dodge=True
                    )
            plt.title(f"{m} — AD vs HC")
            plt.ylabel(f"{m}value")
            plt.xlabel("Frequency band")
            plt.tight_layout()
            plt.show()



            


         

#--------
# TEST
#--------

def covaries(hf, tsv_path="eeg_data/participants.tsv", n_perm=500, seed=42):
    participants = pd.read_csv(tsv_path, sep="\t")
    ad_mask = participants["Group"] == "AD"
    hc_mask = participants["Group"] == "HC"

    # Load data
    sampen = hf["sampen"][:]
    psd = psd_freq(hf["psd"][:])
    plv = hf["plv"][:]
    ge = hf["ge"][:]
    cc = hf["cc"][:]
    cpl = hf["cpl"][:]
    smm = hf["sm"][:]

    # Create group dictionary with covariates
    group = {
        "AD": {
            "sampen": sampen[ad_mask.to_numpy(),:,:],
            "psd": psd[ad_mask.to_numpy(),:,:],
            "plv": mean_plv_upper(plv[ad_mask.to_numpy(),:,:,:]),
            "ge": ge[ad_mask.to_numpy(),:],
            "cc": cc[ad_mask.to_numpy(),:],
            "cpl": cpl[ad_mask.to_numpy(),:],
            "sm": smm[ad_mask.to_numpy(),:],
            "age": participants.loc[ad_mask, "Age"].values,
            "gender": participants.loc[ad_mask, "Gender"].values
        },
        "HC": {
            "sampen": sampen[hc_mask.to_numpy(),:,:],
            "psd": psd[hc_mask.to_numpy(),:,:],
            "plv": mean_plv_upper(plv[hc_mask.to_numpy(),:,:,:]),
            "ge": ge[hc_mask.to_numpy(),:],
            "cc": cc[hc_mask.to_numpy(),:],
            "cpl": cpl[hc_mask.to_numpy(),:],
            "sm": smm[hc_mask.to_numpy(),:],
            "age": participants.loc[hc_mask, "Age"].values,
            "gender": participants.loc[hc_mask, "Gender"].values
        }
    }

    _, n_ch, n_wave = group["AD"]["sampen"].shape
    network_metrics = ["ge", "cc", "cpl", "sm"]
    n_network_measures = group["AD"]["ge"].shape[1]  # assuming all have same number of measures

    # Results dictionary
    results = {
        "sampen_psd": np.zeros((2, n_ch, n_wave, 3)),      # stat, perm_p, effect_size
        "network": np.zeros((4, n_network_measures, 3)),   # ge, cc, cpl, sm
        "plv": np.empty(n_wave, dtype=object),
    }

    # Loop over "sampen" and "psd"
    for i, m in enumerate(["sampen", "psd"]):
        ad_data = group["AD"][m]
        hc_data = group["HC"][m]
        all_labels = ["AD"]*ad_data.shape[0] + ["HC"]*hc_data.shape[0]

        cov_df = pd.DataFrame({
            "age": np.concatenate([group["AD"]["age"], group["HC"]["age"]]),
            "gender": (np.concatenate([group["AD"]["gender"], group["HC"]["gender"]]) == "M").astype(int)
        })

        for ch in range(n_ch):
            for w in range(n_wave):
                vals_ad = ad_data[:, ch, w]
                vals_hc = hc_data[:, ch, w]

                # Cohen's d
                pooled_sd = np.sqrt(
                    ((len(vals_ad)-1)*np.var(vals_ad, ddof=1) + (len(vals_hc)-1)*np.var(vals_hc, ddof=1)) /
                    (len(vals_ad) + len(vals_hc) - 2)
                )
                effect_size_d = (np.mean(vals_ad) - np.mean(vals_hc)) / pooled_sd

                # Permutation ANCOVA
                vals = np.concatenate([vals_ad, vals_hc])
                stat, perm_p = permutation_ancova(vals, all_labels, cov_df, n_perm=n_perm, seed=seed)

                results["sampen_psd"][i, ch, w, 0] = stat
                results["sampen_psd"][i, ch, w, 1] = perm_p
                results["sampen_psd"][i, ch, w, 2] = effect_size_d

    # Loop over network metrics
    for i, m in enumerate(network_metrics):
        ad_data = group["AD"][m]
        hc_data = group["HC"][m]
        all_labels = ["AD"]*ad_data.shape[0] + ["HC"]*hc_data.shape[0]

        cov_df = pd.DataFrame({
            "age": np.concatenate([group["AD"]["age"], group["HC"]["age"]]),
            "gender": (np.concatenate([group["AD"]["gender"], group["HC"]["gender"]]) == "M").astype(int)
        })

        n_measures = ad_data.shape[1]
        for m_idx in range(n_measures):
            vals_ad = ad_data[:, m_idx]
            vals_hc = hc_data[:, m_idx]

            # Cohen's d
            pooled_sd = np.sqrt(
                ((len(vals_ad)-1)*np.var(vals_ad, ddof=1) + (len(vals_hc)-1)*np.var(vals_hc, ddof=1)) /
                (len(vals_ad) + len(vals_hc) - 2)
            )
            effect_size_d = (np.mean(vals_ad) - np.mean(vals_hc)) / pooled_sd

            # Permutation ANCOVA
            vals = np.concatenate([vals_ad, vals_hc])
            stat, perm_p = permutation_ancova(vals, all_labels, cov_df, n_perm=n_perm, seed=seed)

            results["network"][i, m_idx, 0] = stat
            results["network"][i, m_idx, 1] = perm_p
            results["network"][i, m_idx, 2] = effect_size_d


   

    for w in range(n_wave):
        # Extract PLV vectors for this wave
        ad_vals = group["AD"]["plv"][:, w, :]  # shape: (n_AD, 171)
        hc_vals = group["HC"]["plv"][:, w, :]  # shape: (n_HC, 171)
        
        # Compute mean PLV per subject for this wave
        ad_mean = ad_vals.mean(axis=1)
        hc_mean = hc_vals.mean(axis=1)
        
        all_labels = ["AD"]*len(ad_mean) + ["HC"]*len(hc_mean)
        cov_df = pd.DataFrame({
            "age": np.concatenate([group["AD"]["age"], group["HC"]["age"]]),
            "gender": (np.concatenate([group["AD"]["gender"], group["HC"]["gender"]]) == "M").astype(int)
        })

        # Compute pooled SD and Cohen's d
        pooled_sd = np.sqrt(
            ((len(ad_mean)-1)*np.var(ad_mean, ddof=1) + (len(hc_mean)-1)*np.var(hc_mean, ddof=1)) /
            (len(ad_mean) + len(hc_mean) - 2)
        )
        effect_size_d = (np.mean(ad_mean) - np.mean(hc_mean)) / pooled_sd

        # Permutation ANCOVA
        vals_all = np.concatenate([ad_mean, hc_mean])
        stat, perm_p = permutation_ancova(vals_all, all_labels, cov_df, n_perm=n_perm, seed=seed)

        results["plv"][w] = {"stat": stat, "perm_p": perm_p, "effect_size": effect_size_d}
    """
    alpha = 0.05  # significance threshold

    for metric, arr in results.items():
        print(f"\nMetric: {metric}, type: {type(arr)}, shape: {getattr(arr, 'shape', 'N/A')}")
        
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            # Check min, max
            print("  Min:", np.min(arr), "Max:", np.max(arr))
            print("  First element slice:", arr.flat[0])
            
            # If it has perm_p values (4th dimension or last dimension index 1)
            if arr.shape[-1] >= 2:
                perm_p = arr[..., 1]
                sig_mask = perm_p < alpha
                num_sig = np.sum(sig_mask)
                print(f"  Number of significant comparisons (p<{alpha}): {num_sig}")
                
                effect_sizes = arr[..., 2] if arr.shape[-1] >= 3 else None
                if effect_sizes is not None:
                    print("  Effect sizes of significant comparisons:")
                    print(effect_sizes[sig_mask])
        
        elif isinstance(arr, np.ndarray) and arr.dtype == object:
            # For object arrays (like plv)
            for i, val in enumerate(arr):
                if val is None:
                    print(f"  Element {i} is empty")
                else:
                    print(f"  Element {i} filled: {val}")"""


    return( group , results)



#--------------------
# HELPER   FOR TEST
#--------------------

def permutation_ancova(vals, group_labels, covariates, n_perm=500, seed=42):
    np.random.seed(seed)
    df = covariates.copy()
    df["feature"] = vals
    df["group"] = (np.array(group_labels) == "AD").astype(int)

    # Residualize feature wrt covariates
    cov_model = OLS(df["feature"], sm.add_constant(df[["age","gender"]])).fit()
    residuals = cov_model.resid

    # Compute original group t-stat
    group_vals = df["group"]
    obs_stat = np.mean(residuals[group_vals==1]) - np.mean(residuals[group_vals==0])

    # Permutations
    perm_stats = []
    for _ in range(n_perm):
        perm_group = np.random.permutation(group_vals)
        stat = np.mean(residuals[perm_group==1]) - np.mean(residuals[perm_group==0])
        perm_stats.append(stat)

    perm_stats = np.array(perm_stats)
    perm_p = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))

    return obs_stat, perm_p
   
def psd_freq(psd):

      
    bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta":  (12, 30),
            "gamma": (30, 45)
        }

    if psd.ndim==3:  
        n_subj, n_ch, n_bins = psd.shape
        freqs = np.linspace(0.5, 45, n_bins) 
        psd_band = np.zeros((n_subj, n_ch, len(bands)))

        for i, (band, (fmin, fmax)) in enumerate(bands.items()):
            idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            psd_band[:, :, i] = psd[:, :, idx].sum(axis=2)

        
        return(psd_band)
    
    elif psd.ndim==4:
        n_subj,n_epoch, n_ch, n_bins = psd.shape
        freqs = np.linspace(0.5, 45, n_bins) 
        psd_band = np.zeros((n_subj, n_epoch, n_ch, len(bands)))

        for i, (band, (fmin, fmax)) in enumerate(bands.items()):
            idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            psd_band[:, :, :, i] = psd[:, :, :, idx].sum(axis=3)

        return(psd_band)

def mean_plv_upper(plv):
    n_subj, n_waves, n_ch, _ = plv.shape
    n_upper = n_ch * (n_ch - 1) // 2
    upper_plv = np.zeros((n_subj, n_waves, n_upper))

    triu_idx = np.triu_indices(n_ch, k=1)

    for s in range(n_subj):
        for w in range(n_waves):
            # Manually assign upper-triangle values
            count = 0
            for i, j in zip(triu_idx[0], triu_idx[1]):
                upper_plv[s, w, count] = plv[s, w, i, j]
                count += 1

    return upper_plv



#------------------
# HELPER FOR VIZ
#------------------

def to_long(arr, metric_name, group_name, band_names=None, ch_list=None):
    
    rows = []
    arr = np.array(arr)
    
    if arr.ndim == 3:
        n2, n3 = arr.shape[1], arr.shape[2]

        # Case 1: channels x bands (last dim small, e.g., <=5)
        if n3 <= 5:
            for subj in range(arr.shape[0]):
                for ch in range(n2):
                    for b in range(n3):
                        rows.append({
                            "group": group_name,
                            "subject": subj,
                            "metric": metric_name,
                            "channel": ch_list[ch] if ch_list else ch,
                            "connection": None,
                            "band": band_names[b] if band_names and b < len(band_names) else b,
                            "value": arr[subj, ch, b]
                        })
        # Case 2: channels x connections (last dim large, e.g., PLV)
        else:
            # Case 2: channels x connections (PLV)
            for subj in range(arr.shape[0]):
                for ch in range(arr.shape[1]):
                    for conn in range(arr.shape[2]):
                        rows.append({
                            "group": group_name,
                            "subject": subj,
                            "metric": metric_name,
                            "channel": ch_list[ch] if ch_list else ch,
                            "connection": conn,
                            "band": band_names[ch] if band_names and ch < len(band_names) else ch,
                            "value": arr[subj, ch, conn]
                        })
    elif arr.ndim == 2:
        for ch in range(arr.shape[0]):
            for b in range(arr.shape[1]):
                rows.append({
                    "group": group_name,
                    "subject": None,
                    "metric": metric_name,
                    "channel": ch_list[ch] if ch_list and ch < len(ch_list) else ch,
                    "connection": None,
                    "band": band_names[b] if band_names and b < len(band_names) else b,
                    "value": arr[ch, b]
                })
    elif arr.ndim == 1:
        for b in range(arr.shape[0]):
            rows.append({
                "group": group_name,
                "subject": None,
                "metric": metric_name,
                "channel": None,
                "connection": None,
                "band": band_names[b] if band_names and b < len(band_names) else b,
                "value": arr[b]
            })
    return pd.DataFrame(rows)

def stats_to_long(arr, metric_name, stat_type, band_names=None, ch_list=None):
    
    rows = []
    arr = np.array(arr)

    if arr.ndim == 3 and arr.shape[2] == 2:  
        # case: test results with t & p
        for ch in range(arr.shape[0]):
            for b in range(arr.shape[1]):
                rows.append({
                    "metric": metric_name,
                    "channel": ch_list[ch] if ch_list else ch,
                    "band": band_names[b] if band_names and b < len(band_names) else b,
                    "stat": "t",
                    "value": arr[ch, b, 0]
                })
                rows.append({
                    "metric": metric_name,
                    "channel": ch_list[ch] if ch_list else ch,
                    "band": band_names[b] if band_names and b < len(band_names) else b,
                    "stat": "p",
                    "value": arr[ch, b, 1]
                })

    elif arr.ndim == 2:  
        # case: effect sizes (channels × bands or networks × bands)
        for ch in range(arr.shape[0]):
            for b in range(arr.shape[1]):
                rows.append({
                    "metric": metric_name,
                    "channel": ch_list[ch] if ch_list and ch < len(ch_list) else ch,
                    "band": band_names[b] if band_names and b < len(band_names) else b,
                    "stat": stat_type,
                    "value": arr[ch, b]
                })

    elif arr.ndim == 1:  
        # case: network-level (like nbs_wave)
        for b in range(arr.shape[0]):
            rows.append({
                "metric": metric_name,
                "channel": None,
                "band": band_names[b] if band_names and b < len(band_names) else b,
                "stat": stat_type,
                "value": arr[b]
            })

    return pd.DataFrame(rows)



#-----------------
#  OPTIONAL 
#-----------------
def unicovaries_test_avg(hf,tsv_path: str = "eeg_data/participants.tsv"):
    
    participants = pd.read_csv(tsv_path, sep="\t")
    ad_mask = participants["Group"] == "AD"
    hc_mask = participants["Group"] == "HC"
    
    sampen = hf["sampen"][:]
    psd=psd_freq(hf["psd"][:])
    plv=hf["plv"][:]
    ge =hf["ge"][:]
    cc = hf["cc"][:]
    cpl = hf["cpl"][:]
    sm = hf["sm"][:]

    _,n_ch,n_wave = sampen.shape    
     
    tempo={
        "AD":plv[ad_mask],
        "HC":plv[hc_mask]
    }
    groups={
        "AD":{
            "sampen":sampen[ad_mask.to_numpy()],
            "psd":psd[ad_mask.to_numpy()],
            "plv":mean_plv_upper(plv[ad_mask.to_numpy()]),
            "ge":ge[ad_mask],
            "cc":cc[ad_mask],
            "cpl":cpl[ad_mask],
            "sm":sm[ad_mask],
        },
        "HC":{
            "sampen":sampen[hc_mask.to_numpy()],
            "psd":psd[hc_mask.to_numpy()],
            "plv":mean_plv_upper(plv[hc_mask.to_numpy()]),
            "ge":ge[hc_mask],
            "cc":cc[hc_mask],
            "cpl":cpl[hc_mask],
            "sm":sm[hc_mask],
        }}
    
    results = {
    "sampen_psd": np.zeros((2, n_ch, n_wave, 4)),  # 0:sampen, 1:psd, last dim = stat,p_val,effect
    "plv": np.empty((n_wave,2) ,dtype=object),        
    "network": np.zeros((4, n_wave, 4))}          # ge, cc, cpl, sm

  

    metrics = ["sampen", "psd", "plv", "ge", "cc", "cpl", "sm"]

    for i, m in enumerate(metrics):
        ad = groups["AD"][m]
        hc = groups["HC"][m]

        if m in ["sampen", "psd"]:  
            all_p = []

            for ch in range(n_ch):
                for w in range(n_wave):
                    stat, p_value = stats.mannwhitneyu(ad[:, ch, w], hc[:, ch, w], alternative='two-sided')
                    effect_size = 1 - (2 * stat) / (len(ad[:, ch, w]) * len(hc[:, ch, w]))

                    results["sampen_psd"][i, ch, w, 0] = stat
                    results["sampen_psd"][i, ch, w, 2] = effect_size
                    all_p.append(p_value)

            # FDR across all ch×wave
            reject, fdr_p, _, _ = multipletests(all_p, alpha=0.05, method="fdr_bh")

            idx = 0
            for ch in range(n_ch):
                for w in range(n_wave):
                    results["sampen_psd"][i, ch, w, 1] = fdr_p[idx]
                    results["sampen_psd"][i, ch, w, 3] = reject[idx]
                    idx += 1

        elif m in ["ge", "cc", "cpl", "sm"]:  
            all_p = []
            metric_index = ["ge", "cc", "cpl", "sm"].index(m)

            for w in range(n_wave):
                stat, p_value = stats.mannwhitneyu(ad[:, w], hc[:, w], alternative='two-sided')
                effect_size = 1 - (2 * stat) / (len(ad[:, w]) * len(hc[:, w]))

                results["network"][metric_index, w, 0] = stat
                results["network"][metric_index, w, 2] = effect_size
                all_p.append(p_value)

            # FDR across waves
            reject, fdr_p, _, _ = multipletests(all_p, alpha=0.05, method="fdr_bh")

            for w in range(n_wave):
                results["network"][metric_index, w, 1] = fdr_p[w]
                results["network"][metric_index, w, 3] = reject[w]

        elif m == "plv":
            pass

    


        

    for w in range(n_wave):
    
        AD = np.transpose(tempo["AD"][:,w,:,:],(1, 2, 0))
        HC = np.transpose(tempo["HC"][:,w,:,:],(1, 2, 0))
        pval, compo,_ = bct.nbs_bct(AD,HC, thresh=2.5,tail='both',k=1000,seed=42)
        results["plv"][w,0] = pval
        results["plv"][w,1] = compo


    return (groups , results)
     

     

if __name__ == "__main__":
    main()