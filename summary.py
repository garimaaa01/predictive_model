# summary.py
#
# run:
#   python summary.py split        - single-split pipeline: one train/test
#                                     split, covaries() run once on everyone
#                                     before the split. this design leaks
#                                     info from test subjects into feature
#                                     selection, so its accuracy number
#                                     isn't a trustworthy estimate.
#   python summary.py kfold        - k-fold cross-validation: covaries()
#                                     reruns fresh per fold, on training
#                                     subjects only. no leakage. EPV=10
#                                     default.
#   python summary.py kfold <epv>  - k-fold, custom EPV
#                                     e.g. python summary.py kfold 5
#
from test import main, covaries, psd_freq, mean_plv_upper
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import permutations
import pandas as pd
import seaborn as sns
import h5py
import os

# fixed output folder for this file's statistical report figures - each
# rerun overwrites the same filenames instead of piling up new ones
SUMMARY_VIZ_DIR = "results/summary_viz"
os.makedirs(SUMMARY_VIZ_DIR, exist_ok=True)





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


#---------------------------------------
# CROSS-VALIDATION (leakage fix)
#---------------------------------------

def extract_features_for_subjects(hf, feature_map, subject_indices):
    """
    builds a feature matrix for ANY set of subjects (e.g. a fold's
    held-out test subjects), reading columns straight from data.h5 in the
    exact order given by feature_map - same feature_map that fold's
    training run of prepare_lasso_data() produced. lets held-out subjects
    get scored on exactly the features LASSO trained on, without ever
    passing through covaries() or touching feature selection.

    subject_indices: row positions into participants.tsv / data.h5 arrays
    (not split by AD/HC - order matches feature_map, each row here = one
    subject_indices entry).
    """
    sampen = hf["sampen"][:][subject_indices]
    psd = psd_freq(hf["psd"][:])[subject_indices]
    plv = mean_plv_upper(hf["plv"][:])[subject_indices]
    ge = hf["ge"][:][subject_indices]
    cc = hf["cc"][:][subject_indices]
    cpl = hf["cpl"][:][subject_indices]
    sm = hf["sm"][:][subject_indices]

    metric_arrays = {"sampen": sampen, "psd": psd, "ge": ge, "cc": cc, "cpl": cpl, "sm": sm}

    n_subjects = len(subject_indices)
    n_features = len(feature_map)
    X = np.zeros((n_subjects, n_features))

    for col, f in enumerate(feature_map):
        metric = f[0]
        if metric in ["sampen", "psd"]:
            _, ch, wave = f
            X[:, col] = metric_arrays[metric][:, ch, wave]
        elif metric in ["ge", "cc", "cpl", "sm"]:
            _, wave = f
            X[:, col] = metric_arrays[metric][:, wave]
        elif metric == "plv":
            _, wave, edge = f
            X[:, col] = plv[:, wave, edge]

    return X


def cap_features_by_effect_size(result, sig_indices, max_features):
    """
    quasi-separation root cause fix: restricting C alone couldn't fully
    kill the divide-by-zero/overflow warnings in high-feature folds,
    since the inner CV search was itself vulnerable to the same
    small-sample overfitting it was supposed to guard against. actual
    cause - too many features relative to subjects - so keep only the top
    `max_features`, ranked by ABSOLUTE effect size (Cohen's d), across all
    metrics combined.

    effect size, not p-value, is the ranking criterion: p-value reflects
    both effect size AND sample size/noise, so ranking by p in a small,
    uneven fold sample can favor features significant mostly from a lucky
    draw rather than real separation. effect size is more direct.

    cap needs to come from a sample-size rule BEFORE seeing what accuracy
    it produces - not tuned after the fact to chase a better number.
    """
    all_candidates = []  # (metric, index_tuple, abs_effect_size)

    for ch, w in sig_indices.get("sampen", []):
        eff = result["sampen_psd"][0, ch, w, 2]
        all_candidates.append(("sampen", (ch, w), abs(eff)))
    for ch, w in sig_indices.get("psd", []):
        eff = result["sampen_psd"][1, ch, w, 2]
        all_candidates.append(("psd", (ch, w), abs(eff)))
    for i, metric in enumerate(["ge", "cc", "cpl", "sm"]):
        for w in sig_indices.get(metric, []):
            eff = result["network"][i, w, 2]
            all_candidates.append((metric, (w,), abs(eff)))
    for w, edge in sig_indices.get("plv", []):
        eff = result["plv"][w, edge, 2]
        all_candidates.append(("plv", (w, edge), abs(eff)))

    all_candidates.sort(key=lambda x: -x[2])
    kept = all_candidates[:max_features]

    capped_sig_indices = {"sampen": [], "psd": [], "ge": [], "cc": [], "cpl": [], "sm": [], "plv": []}
    for metric, idx_tuple, _ in kept:
        if metric in ["sampen", "psd"]:
            capped_sig_indices[metric].append(idx_tuple)
        elif metric in ["ge", "cc", "cpl", "sm"]:
            capped_sig_indices[metric].append(idx_tuple[0])
        elif metric == "plv":
            capped_sig_indices[metric].append(idx_tuple)

    return capped_sig_indices, len(all_candidates)


def run_cross_validation(hf, tsv_path="eeg_data/participants.tsv", n_folds=5, alpha=0.05, seed=42,
                          epv=10):
    """
    leakage fix: replaces single 70/30 split with k-fold CV where feature
    selection (covaries) reruns FRESH inside each fold, using ONLY that
    fold's training subjects. before this, covaries() ran once on the
    whole dataset before any split existed, so the "significant" features
    LASSO got tested on were already chosen using info from subjects that
    became the test set - the leakage behind the original inflated 80%.
    now each fold's held-out subjects stay invisible to both covaries()
    and LASSO training until the single moment they get scored.

    quasi-separation root cause fix: epv (events per variable) sets the
    MAX features allowed per fold, as
    max_features = (smaller class's training count) // epv. standard rule
    of thumb in logistic regression lit for how many predictors a sample
    can support without serious overfitting/separation risk (commonly
    cited: ~10 events per variable minimum). ratio decided by sample size
    ALONE, before any fold runs, before any accuracy is seen - not tuned
    after to chase a better-looking result. features beyond the cap get
    dropped weakest-effect-size-first (see cap_features_by_effect_size).
    pre-cap and post-cap counts both printed per fold for transparency.
    """
    participants = pd.read_csv(tsv_path, sep="\t")
    labels = (participants["Group"] == "AD").astype(int).values  # 1=AD, 0=HC, for stratification only
    all_indices = np.arange(len(participants))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accuracies = []
    fold_n_features = []
    selected_feature_counts = {}  # tracks how often each feature is selected across folds
    selected_feature_coefs = {}   # tracks each fold's coefficient for features that got selected

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(all_indices, labels), start=1):

        # feature selection using ONLY this fold's training subjects
        group_train, result_train = covaries(hf, tsv_path=tsv_path, subject_indices=train_idx, seed=seed)
        sig_indices_uncapped = get_significant_indices(result_train, alpha=alpha)

        # cap decided from training class sizes alone, before this fold's
        # accuracy is known
        n_ad_train = int(labels[train_idx].sum())
        n_hc_train = len(train_idx) - n_ad_train
        max_features = max(1, min(n_ad_train, n_hc_train) // epv)

        sig_indices, n_uncapped = cap_features_by_effect_size(result_train, sig_indices_uncapped, max_features)
        print(f"Fold {fold_num}: {n_uncapped} significant features found, "
              f"capped to top {max_features} by |effect size| "
              f"(EPV rule: min({n_ad_train},{n_hc_train})//{epv})")
        filtered_group_train = filtering_groups(group_train, sig_indices)

        X_train, y_train, feature_map = prepare_lasso_data(filtered_group_train)

        if len(feature_map) == 0:
            print(f"Fold {fold_num}: no significant features found, skipping.")
            continue

        # diagnostic: near-zero-variance columns - candidate cause of
        # divide-by-zero warnings (StandardScaler dividing by ~0 std ->
        # huge/inf scaled values)
        col_std = X_train.std(axis=0)
        near_zero_var_cols = np.where(col_std < 1e-8)[0]
        if len(near_zero_var_cols) > 0:
            print(f"Fold {fold_num}: WARNING - {len(near_zero_var_cols)} near-zero-variance "
                  f"feature(s) found: {[feature_map[i] for i in near_zero_var_cols]}")

        # diagnostic: feature-to-subject ratio - candidate cause, too many
        # features relative to training subjects -> quasi-complete
        # separation, coefficients pushed to extremes
        ratio = len(feature_map) / X_train.shape[0]
        print(f"Fold {fold_num}: {len(feature_map)} features / {X_train.shape[0]} "
              f"training subjects = {ratio:.2f} features-per-subject "
              f"{'(high -- risk of quasi-separation)' if ratio > 0.5 else ''}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # diagnostic: confirm scaling itself didn't already produce
        # inf/nan (would confirm the zero-variance candidate directly)
        if not np.all(np.isfinite(X_train_scaled)):
            n_bad = np.sum(~np.isfinite(X_train_scaled))
            print(f"Fold {fold_num}: WARNING - {n_bad} non-finite value(s) in scaled "
                  f"training data (inf/nan) -- likely cause of the matmul warnings.")

        # quasi-separation fix: fixed C=1.0 let LASSO chase near-perfect
        # separation in high-feature folds (96, 175 features vs ~52
        # training subjects), coefficients pushed toward infinity ->
        # divide-by-zero/overflow warnings, a sign of overfitting to this
        # fold's specific subjects rather than real generalizable signal.
        # LogisticRegressionCV searches a range of C via an INNER CV split
        # of training data only (never touches this fold's held-out test
        # subjects), picks whatever C generalizes best within training.
        # each fold gets its own appropriately-strong regularization
        # instead of one guessed value everywhere.
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        lasso = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=inner_cv,
            penalty='l1',
            solver='saga',
            max_iter=5000,
            scoring='accuracy',
            random_state=seed,
        )
        lasso.fit(X_train_scaled, y_train)
        print(f"Fold {fold_num}: selected C = {lasso.C_[0]:.4f} (via inner cross-validation)")

        # held-out fold's test data, same features, read fresh from
        # data.h5 - these subjects never touched covaries() or lasso.fit()
        # above
        X_test = extract_features_for_subjects(hf, feature_map, test_idx)
        X_test_scaled = scaler.transform(X_test)
        y_test = (participants.loc[test_idx, "Group"] == "AD").astype(int).values

        # prepare_lasso_data's label_map is {"AD": 0, "HC": 1} - opposite
        # of the 1=AD convention used for y_test above. flip y_test to
        # match lasso's training label convention.
        y_test_lasso_convention = 1 - y_test

        y_pred = lasso.predict(X_test_scaled)
        acc = accuracy_score(y_test_lasso_convention, y_pred)

        fold_accuracies.append(acc)
        fold_n_features.append(len(feature_map))

        for f, coef in zip(feature_map, lasso.coef_[0]):
            if coef != 0:
                selected_feature_counts[f] = selected_feature_counts.get(f, 0) + 1
                selected_feature_coefs.setdefault(f, []).append(coef)

        print(f"Fold {fold_num}: {len(feature_map)} significant features, accuracy = {acc:.3f}")

    fold_accuracies = np.array(fold_accuracies)
    print()
    print("-------------------------------")
    print(f"Mean accuracy across {len(fold_accuracies)} folds: {fold_accuracies.mean():.3f} +/- {fold_accuracies.std():.3f}")
    print(f"Per-fold accuracies: {[f'{a:.3f}' for a in fold_accuracies]}")
    print(f"Per-fold significant feature counts: {fold_n_features}")
    print("-------------------------------")
    print()
    print("Features selected by LASSO in multiple folds (feature: count):")
    for f, count in sorted(selected_feature_counts.items(), key=lambda x: -x[1]):
        if count > 1:
            print(f"  {f}: {count}/{len(fold_accuracies)} folds")

    # plots for the kfold pipeline, not the split one - plot_lasso_channel_band
    # and plot_lasso_plv both just need a dict with "coefficients" and
    # "feature_map" keys, same shape a single lasso_dict has, so reuse them
    # here instead of writing new heatmap code. coefficient per feature =
    # mean coefficient across only the folds that selected it (folds that
    # didn't select it don't get averaged in as a 0 - that would understate
    # how strong the feature is when it IS picked).
    cv_feature_map = list(selected_feature_coefs.keys())
    cv_mean_coefs = np.array([np.mean(selected_feature_coefs[f]) for f in cv_feature_map])
    cv_lasso_dict = {"feature_map": cv_feature_map, "coefficients": [cv_mean_coefs]}

    plot_lasso_channel_band(cv_lasso_dict, metric="sampen")
    plot_lasso_channel_band(cv_lasso_dict, metric="psd")
    plot_lasso_plv(cv_lasso_dict)

    return fold_accuracies, selected_feature_counts

   
  


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

        # pre-existing bug, not from my changes: feature_map used to get
        # appended on EVERY group's pass through this loop. AD and HC
        # always share identical feature keys (both built from the same
        # sig_indices in filtering_groups), so this silently duplicated
        # every entry - feature_map ended up 2x X's actual column count.
        # never crashed before since evaluation_lasso's
        # zip(feature_map, coefficients[0]) just silently truncated to
        # the shorter list. surfaces now because
        # extract_features_for_subjects (used by run_cross_validation)
        # relies on len(feature_map) matching X's real column count.
        # fix: only append to feature_map on the FIRST group's pass.
        build_feature_map = (grp == groups[0])

        # Loop over metrics
        for metric, data in filtered_group[grp].items():
            if metric in ["sampen", "psd"]:
                for (ch, wave), values in data.items():
                    for i, val in enumerate(values):
                        X_list_grp[i].append(val)
                    if build_feature_map:
                        feature_map.append((metric, ch, wave))

            elif metric in ["ge", "cc", "cpl", "sm"]:
                for wave, values in data.items():
                    for i, val in enumerate(values):
                        X_list_grp[i].append(val)
                    if build_feature_map:
                        feature_map.append((metric, wave))

            elif metric == "plv":
                # filtered_group["AD"/"HC"]["plv"] now keyed by
                # (wave, edge) -> 1D array of subject values, one sig
                # connection at a time (see filtering_groups). replaces
                # old branch that expected whole wave-slices (2D
                # subject x edges) and flattened them.
                for (wave, edge), values in data.items():
                    for i, val in enumerate(values):
                        X_list_grp[i].append(val)
                    if build_feature_map:
                        feature_map.append((metric, wave, edge))

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
    # index 3 = FDR-corrected p (see get_significant_indices for why).
    # index 1 = raw p, index 2 = effect size, both unchanged.
    for metric in ["sampen", "psd"]:
        filtered_results[metric] = []
        for ch, w in sig_indices.get(metric, []):
            stat = result["sampen_psd"][["sampen","psd"].index(metric), ch, w, 0]
            p_val = result["sampen_psd"][["sampen","psd"].index(metric), ch, w, 3]
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
            p_val = result["network"][["ge","cc","cpl","sm"].index(metric), w, 3]
            effect_size = result["network"][["ge","cc","cpl","sm"].index(metric), w, 2]
            filtered_results[metric].append({
                "wave": w,
                "stat": stat,
                "p": p_val,
                "effect_size": effect_size
            })

    # --- PLV ---
    # result["plv"] is now a plain array shaped (n_wave, n_edges, 4) -
    # (stat, raw_p, effect_size, fdr_p) - not a list of per-wave dicts.
    # sig_indices["plv"] is now a list of (wave, edge) pairs (see
    # get_significant_indices), so index the array directly instead of
    # dict-style lookups like result["plv"][w]["perm_p"].
    filtered_results["plv"] = []
    for w, edge in sig_indices.get("plv", []):
        stat = result["plv"][w, edge, 0]
        p_val = result["plv"][w, edge, 3]
        effect_size = result["plv"][w, edge, 2]
        filtered_results["plv"].append({
            "wave": w,
            "edge": edge,
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
            # indices now a list of (wave, edge) pairs - one entry per
            # individual significant connection - instead of whole-wave
            # numbers. old version grabbed the entire wave's 171-connection
            # slice (group["AD"]["plv"][:, w, :]) for every significant
            # wave. now grabs only the ONE connection that was itself
            # found significant (group["AD"]["plv"][:, w, edge]), keyed by
            # (wave, edge) so prepare_lasso_data can tell connections apart.
            filtered_group["AD"][metric] = {}
            filtered_group["HC"][metric] = {}
            for w, edge in indices:
                filtered_group["AD"][metric][(w, edge)] = group["AD"]["plv"][:, w, edge]
                filtered_group["HC"][metric][(w, edge)] = group["HC"]["plv"][:, w, edge]
    
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
        "plv": [(wave, edge), ...]
    }
    """
    sig_indices = {}

    # ----------------- SampEn and PSD -----------------
    # index 3 = FDR-corrected p now. covaries() used to only store
    # (stat, raw_p, effect_size) - raw_p at index 1 - no correction at
    # all. now stores (stat, raw_p, effect_size, fdr_p), so "significant"
    # is decided on the corrected p, not raw - otherwise the FDR fix in
    # test.py has no effect here.
    res_sampen_psd = result["sampen_psd"]
    metrics = ["sampen", "psd"]
    for i, metric_name in enumerate(metrics):
        fdr_p = res_sampen_psd[i, :, :, 3]  # FDR-corrected p-values
        idx = np.argwhere(fdr_p < alpha)
        sig_indices[metric_name] = [tuple(x) for x in idx]  # (ch, wave)

    # ----------------- Network metrics -----------------
    network_metrics = ["ge", "cc", "cpl", "sm"]
    res_network = result["network"]
    for i, metric_name in enumerate(network_metrics):
        fdr_p = res_network[i, :, 3]  # FDR-corrected p-values per measure
        waves = np.where(fdr_p < alpha)[0]
        sig_indices[metric_name] = list(waves)

    # ----------------- PLV -----------------
    # result["plv"] now a plain array shaped (n_wave, n_edges, 4) -
    # (stat, raw_p, effect_size, fdr_p) - since test.py tests every
    # individual connection separately now instead of one average per
    # wave. old version looped over per-wave dict objects
    # (res_plv[w]["perm_p"]); now checks each (wave, edge) connection's
    # own FDR-corrected p directly against the array.
    res_plv = result["plv"]
    n_wave, n_edges, _ = res_plv.shape
    plv_sig = []
    for w in range(n_wave):
        for edge in range(n_edges):
            if res_plv[w, edge, 3] < alpha:
                plv_sig.append((w, edge))
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
        plt.savefig(f"{SUMMARY_VIZ_DIR}/lasso_{metric}_heatmap.png", dpi=200, bbox_inches='tight')
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
    plt.savefig(f"{SUMMARY_VIZ_DIR}/lasso_plv_heatmap.png", dpi=200, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    import sys
    # explicit choice required now - "split" for the single-split design,
    # "kfold" for the cross-validation design. no more no-arg-means-split
    # default, since that made it too easy to run the leakage-prone
    # design by accident.
    if len(sys.argv) >= 2 and sys.argv[1] == "kfold":
        # optional 3rd arg sets EPV cap ratio, e.g.
        # "python summary.py kfold 5" for EPV=5. defaults to 10.
        epv = int(sys.argv[2]) if len(sys.argv) == 3 else 10
        with h5py.File("data.h5", "r") as hf:
            run_cross_validation(hf, epv=epv)
    elif len(sys.argv) == 2 and sys.argv[1] == "split":
        # single-split design, kept for comparison - covaries() runs on
        # everyone before the split happens, which leaks test-subject info
        # into feature selection, so this accuracy number isn't the
        # trustworthy one.
        summary()
    else:
        print("Usage: python summary.py split        (single train/test split)")
        print("       python summary.py kfold        (k-fold cross-validation, EPV=10)")
        print("       python summary.py kfold <epv>  (k-fold cross-validation, custom EPV)")