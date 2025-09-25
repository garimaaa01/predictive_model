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
```bash
git clone https://github.com/garimaaa01/predictive_model.git
cd predictive_model
