from test import main
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import permutations
import pandas as pd
import seaborn as sns





def summary():
   
    " loading the data  to get the significant values"
    group, result = main()
    alpha = 0.05
    sig_indices = get_significant_indices(result, alpha=alpha)
    
    filtered_results = filtering_results(result, sig_indices)
    filtered_group = filtering_groups(group, sig_indices)
  
    " the lasso prediction "
    lasso_dictionary,data_train ,data_label_train = lasso_prediction(filtered_group)
    selected_features = evaluation_lasso(lasso_dictionary,data_train ,data_label_train)
    # (optional) top_triplets_per_subject= detailed_factor(selected_features, group)

   


    plot_lasso_channel_band(lasso_dictionary, metric="sampen")
    plot_lasso_channel_band(lasso_dictionary, metric="psd")
    plot_lasso_plv(lasso_dictionary)

   
  

      

     
   





#--------
# Lasso 
#--------

def evaluation_lasso(lasso_dictionary,data_train ,data_label_train):
 




    # Test set accuracy
    accuracy = accuracy_score(lasso_dictionary["data_label_test"],
                          lasso_dictionary["model"].predict(lasso_dictionary["data_test"]))
    
    print()
    print("-------------------------------")
    print(f" the sparsity is {lasso_dictionary["sparsity"]}")
    print("-------------------------------")
    print()
    print("-------------------------------")
    print(f" the intercept is {(lasso_dictionary["intercept"])}")
    print("-------------------------------")
    print()
    print("-------------------------------")
    print("Test set accuracy:", accuracy) 
    print("-------------------------------")

   
 
    selected_features = []

    for i, (f, coef) in enumerate(zip(lasso_dictionary["feature_map"], lasso_dictionary["coefficients"][0])):
       if coef !=0:
            selected_features.append((f, coef))
            mean_hc = data_train[data_label_train == 0, i].mean()
            mean_ad = data_train[data_label_train == 1, i].mean()

            direction = "AD" if coef > 0 else "HC"
            print(f"Feature: {f}, Coef: {coef:.3f}, Mean HC: {mean_hc:.3f}, Mean AD: {mean_ad:.3f}, Predicts: {direction}")

    return selected_features

"optional if want to see individual's top 10 significant factor "
def detailed_factor(selected_features, group ,tsv_path="eeg_data/participants.tsv"):
    #------------------------------------------------
    # PERMUTATION INORDER TO FIND OUT  TIPPING POINT 
    #------------------------------------------------

    participants = pd.read_csv(tsv_path, sep="\t")
    ad_mask = participants["Group"] == "AD"
    hc_mask = participants["Group"] == "HC"
    mmse = participants["MMSE"].to_list()

    sig_indices = {"sampen":[],
                "psd":[],
                "plv":[],
                "ge":[],
                "cc":[],
                "cpl":[],
                "sm":[]  
                }
    for each in selected_features:    
        f = each[0]    
        sig_indices[f[0]].append((f[1],f[2]))
            
    selected_group = filtering_groups(group,sig_indices)
    data, data_label , feature_names = flatten_selected_group(selected_group)

    

    # Fix feature names mismatch
    n_subjects, n_features = data.shape
    feature_names = feature_names[:n_features]  # only first 180 names

    # Training  predictive model on full dataset
    model = LogisticRegression(max_iter=1000)
    model.fit(data, data_label)

    #  top_k features for efficiency
    top_k = 50
    top_feature_indices = np.arange(top_k)  # using first top_k features
    triplets = np.array(list(permutations(top_feature_indices, 3)))  # shape: (num_triplets, 3)

    num_triplets = triplets.shape[0]

    top_triplets_per_subject = []

    for i in range(n_subjects):
        # Fully vectorized triplet feature matrix
        X_triplets = np.zeros((num_triplets, n_features))
        
        # Fancy indexing to fill triplet values without Python loop
        rows = np.arange(num_triplets)[:, None]   # shape: (num_triplets, 1)
        cols = triplets                            # shape: (num_triplets, 3)
        X_triplets[rows, cols] = data[i, cols]
        
        # Vectorized prediction
        probs = model.predict_proba(X_triplets)[:, 1]  # probability of the event
        
        # Get top 10 triplets efficiently
        top_idx = np.argpartition(-probs, 10)[:10]
        top_probs = probs[top_idx]
        top_triplets = triplets[top_idx]
        
        # Sort top 10 descending
        sort_idx = np.argsort(-top_probs)
        top_probs = top_probs[sort_idx]
        top_triplets = top_triplets[sort_idx]
        
        # Save results per subject
        top_triplets_per_subject.append([
        {
            'indices': triplet,
            'names': [feature_names[x] for x in triplet],
            'values': [data[i, x] for x in triplet],
            'score': score,      # use the score from top_probs
            'label': data_label[i]
        } for triplet, score in zip(top_triplets, top_probs)
    ])
        

    return (top_triplets_per_subject)
      

def flatten_selected_group(selected_group):
    feature_names = []
    X_list = []
    y_list = []

    for grp, label in zip(["AD", "HC"], [0, 1]):
            n_subjects = None

            # Determine number of subjects from the first valid feature
            for metric, features in selected_group[grp].items():
                for feat_key, values in features.items():
                    if isinstance(values, np.ndarray):
                        n_subjects = values.shape[0]
                        break
                if n_subjects is not None:
                    break

            if n_subjects is None:
                continue

            # pre-allocate subject rows
            X_grp = [[] for _ in range(n_subjects)]

            for metric, features in selected_group[grp].items():
                for feat_key, values in features.items():
                    if not isinstance(values, np.ndarray):
                        # skip non-array features like effect_size
                        continue

                    # handle plv: 2D array (subject x edges)
                    if values.ndim == 2:
                        for i in range(n_subjects):
                            X_grp[i].extend(values[i, :])
                        for edge_idx in range(values.shape[1]):
                            feature_names.append((metric, feat_key, edge_idx))
                    # handle 1D array: SampEn, PSD
                    elif values.ndim == 1:
                        for i in range(n_subjects):
                            X_grp[i].append(values[i])
                        feature_names.append((metric, feat_key))
                    else:
                        raise ValueError(f"Unexpected array shape for {metric} {feat_key}: {values.shape}")

            X_list.extend(X_grp)
            y_list.extend([label] * n_subjects)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, feature_names


def lasso_prediction(filtered_group):

    data, data_label, feature_map = prepare_lasso_data(filtered_group)

    #-----------------------------------------
    # Train/test split and standardizing data 
    #-----------------------------------------

    data_train ,data_test ,data_label_train , data_label_test = train_test_split(data , data_label, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    #------------------------------------
    # Fitting  LASSO logistic regression
    #------------------------------------
    lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, C=1.0)
    lasso.fit(data_train, data_label_train)

    return ( {
        "scaler":scaler,
        "model":lasso,
        "feature_map":feature_map,
        "data_test":data_test,
        "data_label_test":data_label_test,
        "coefficients":lasso.coef_,
        "sparsity": np.mean(lasso.coef_[0] == 0),
        "intercept": lasso.intercept_[0]

    }, data_train, data_label_train)





#-------------------------------
# Flattening the data for lasso 
#-------------------------------

def prepare_lasso_data(filtered_group):
    """
    Converts filtered_group dict into X (features) and y (labels) suitable for LASSO.
    Also returns feature_map to track which metric/ch/wave each feature corresponds to.
    """
    X_list = []
    y_list = []
    feature_map = []

    groups = ["AD", "HC"]
    label_map = {"AD": 0, "HC": 1}

    for grp in groups:
        # Determine number of subjects from first non-empty metric
        n_subjects = None
        for metric, data in filtered_group[grp].items():
            if len(data) > 0:
                first_key = list(data.keys())[0]
                values = data[first_key]
                n_subjects = values.shape[0]
                break
        if n_subjects is None:
            raise ValueError(f"No data found for group {grp}")

        # Pre-allocate X_list for this group
        X_list_grp = [[] for _ in range(n_subjects)]

        # Loop over metrics
        for metric, data in filtered_group[grp].items():
            if metric in ["sampen", "psd"]:
                for (ch, wave), values in data.items():
                    for i, val in enumerate(values):
                        X_list_grp[i].append(val)
                    feature_map.append((metric, ch, wave))

            elif metric in ["ge", "cc", "cpl", "sm"]:
                for wave, values in data.items():
                    for i, val in enumerate(values):
                        X_list_grp[i].append(val)
                    feature_map.append((metric, wave))

            elif metric == "plv":
                for key, values in data.items():
                    if isinstance(key, tuple) and key[1] == "effect_size":
                        continue
                    flat_vals = values.reshape(values.shape[0], -1)
                    for i in range(values.shape[0]):
                        X_list_grp[i].extend(flat_vals[i])
                    # Track mapping
                    for f in range(flat_vals.shape[1]):
                        feature_map.append((metric, key, f))

        # Add labels
        y_list.extend([label_map[grp]] * n_subjects)
        X_list.extend(X_list_grp)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, feature_map




#--------------------
# Filtering the data 
#--------------------

def filtering_results(result , sig_indices):
    filtered_results = {}

    # --- SampEn and PSD ---
    for metric in ["sampen", "psd"]:
        filtered_results[metric] = []
        for ch, w in sig_indices.get(metric, []):
            stat = result["sampen_psd"][["sampen","psd"].index(metric), ch, w, 0]
            p_val = result["sampen_psd"][["sampen","psd"].index(metric), ch, w, 1]
            effect_size = result["sampen_psd"][["sampen","psd"].index(metric), ch, w, 2]
            filtered_results[metric].append({
                "ch": ch,
                "wave": w,
                "stat": stat,
                "p": p_val,
                "effect_size": effect_size
            })

    # --- Network metrics ---
    for metric in ["ge", "cc", "cpl", "sm"]:
        filtered_results[metric] = []
        for w in sig_indices.get(metric, []):
            stat = result["network"][["ge","cc","cpl","sm"].index(metric), w, 0]
            p_val = result["network"][["ge","cc","cpl","sm"].index(metric), w, 1]
            effect_size = result["network"][["ge","cc","cpl","sm"].index(metric), w, 2]
            filtered_results[metric].append({
                "wave": w,
                "stat": stat,
                "p": p_val,
                "effect_size": effect_size
            })

    # --- PLV ---
    filtered_results["plv"] = []
    for w, eff in sig_indices.get("plv", []):
        stat = result["plv"][w]["stat"]
        p_val = result["plv"][w]["perm_p"]
        effect_size = result["plv"][w]["effect_size"]
        filtered_results["plv"].append({
            "wave": w,
            "stat": stat,
            "p": p_val,
            "effect_size": effect_size
        })
    return(filtered_results)
  

def filtering_groups(group, sig_indices):
    filtered_group = {"AD": {}, "HC": {}}

    for metric, indices in sig_indices.items():

        if metric in ["sampen", "psd"]:
            # Initialize metric dicts
            filtered_group["AD"][metric] = {}
            filtered_group["HC"][metric] = {}
            for ch, w in indices:
                filtered_group["AD"][metric][(ch, w)] = group["AD"][metric][:, ch, w]
                filtered_group["HC"][metric][(ch, w)] = group["HC"][metric][:, ch, w]

        elif metric in ["ge", "cc", "cpl", "sm"]:
            filtered_group["AD"][metric] = {}
            filtered_group["HC"][metric] = {}
            for w in indices:
                filtered_group["AD"][metric][w] = group["AD"][metric][:, w]
                filtered_group["HC"][metric][w] = group["HC"][metric][:, w]

        elif metric == "plv":
            filtered_group["AD"][metric] = {}
            filtered_group["HC"][metric] = {}
            for w, eff in indices:
                filtered_group["AD"][metric][w] = group["AD"]["plv"][:, w, :]
                filtered_group["HC"][metric][w] = group["HC"]["plv"][:, w, :]
                # Optionally store effect size too
                filtered_group["AD"][metric][(w, "effect_size")] = eff
                filtered_group["HC"][metric][(w, "effect_size")] = eff
    
    return(filtered_group)




#-------------------------------------------
# Flag significant pval and ch/waves inedx
#-------------------------------------------

def get_significant_indices(result, alpha=0.05):
    """
   
    Returns a dict:
    {
        "sampen": [(ch, wave), ...],
        "psd": [(ch, wave), ...],
        "ge": [wave_idx, ...],
        "cc": [wave_idx, ...],
        "cpl": [wave_idx, ...],
        "sm": [wave_idx, ...],
        "plv": [(wave, effect_size), ...]
    }
    """
    sig_indices = {}

    # ----------------- SampEn and PSD -----------------
    res_sampen_psd = result["sampen_psd"]
    metrics = ["sampen", "psd"]
    for i, metric_name in enumerate(metrics):
        perm_p = res_sampen_psd[i, :, :, 1]  # p-values
        idx = np.argwhere(perm_p < alpha)
        sig_indices[metric_name] = [tuple(x) for x in idx]  # (ch, wave)

    # ----------------- Network metrics -----------------
    network_metrics = ["ge", "cc", "cpl", "sm"]
    res_network = result["network"]
    for i, metric_name in enumerate(network_metrics):
        perm_p = res_network[i, :, 1]  # p-values per wave
        waves = np.where(perm_p < alpha)[0]
        sig_indices[metric_name] = list(waves)

    # ----------------- PLV -----------------
    res_plv = result["plv"]
    plv_sig = []
    for w, val in enumerate(res_plv):
        if val["perm_p"] < alpha:
            plv_sig.append((w, val["effect_size"]))
    sig_indices["plv"] = plv_sig

    return sig_indices



#----------------
# Visualization 
#----------------
def plot_lasso_channel_band(lasso_dict, metric=''):
        coefs = lasso_dict["coefficients"][0]
        fmap = lasso_dict["feature_map"]

        # Extract only sampen/psd features
        feat = [(f, c) for f, c in zip(fmap, coefs) if f[0] == metric and c != 0]

        if not feat:
            print(f"No {metric} features selected.")
            return

        n_ch = max(f[1] for f, _ in feat) + 1
        n_wave = max(f[2] for f, _ in feat) + 1
        mat = np.zeros((n_ch, n_wave))

        for (m, ch, w), c in feat:
            mat[ch, w] = c

        plt.figure(figsize=(8,6))
        sns.heatmap(mat, cmap="bwr", center=0, cbar_kws={'label': 'Coefficient'})
        plt.xlabel("Wave index")
        plt.ylabel("Channel index")
        plt.title(f"LASSO coefficients — {metric}")
        plt.show()


def plot_lasso_plv(lasso_dict):
    coefs = lasso_dict["coefficients"][0]
    fmap = lasso_dict["feature_map"]

    feat = [(f, c) for f, c in zip(fmap, coefs) if f[0] == "plv" and c != 0]
    if not feat:
        print("No PLV features selected.")
        return

    n_edge = max(f[2] for f, _ in feat) + 1
    n_wave = max(f[1] for f, _ in feat) + 1
    mat = np.zeros((n_edge, n_wave))

    for (m, w, e), c in feat:
        mat[e, w] = c

    plt.figure(figsize=(10,6))
    sns.heatmap(mat, cmap="bwr", center=0, cbar_kws={'label': 'Coefficient'})
    plt.xlabel("Wave index")
    plt.ylabel("Edge index")
    plt.title("LASSO coefficients — PLV")
    plt.show()




if __name__ == "__main__":
    summary()