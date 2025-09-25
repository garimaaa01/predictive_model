# Predictive EEG Signatures of Alzheimer's Disease

This project analyzes EEG data to identify markers that distinguish Alzheimer’s disease (AD) from healthy controls (HC). 

## Features
- Extracts EEG metrics including:
  - Signal complexity (Sample Entropy, SampEn)
  - Spectral power (Power Spectral Density, PSD)
  - Functional connectivity (Phase Locking Value, PLV)
  - Network measures (Global Efficiency, Clustering Coefficient, Characteristic Path Length, Small-Worldness)
- Applies permutation based ANCOVA filtering to identify significant features
- Uses LASSO regression for feature selection and predictive modeling


  
## Installation 
1. Clone the repository:
2. Since this repository is private, you will need to authenticate either using a personal access token (PAT) with HTTPS or via SSH:

```bash
git clone https://github.com/garimaaa01/predictive_model.git
cd predictive_model
```
 Git will ask for username -> enter GitHub username
 Git will ask for password -> enter the personal access token here



3.Download the necessary libraries:
```bash
pip install -r requirements.txt
```

