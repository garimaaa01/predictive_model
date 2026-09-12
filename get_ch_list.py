import pickle

with open("all_epochs.pkl", "rb") as f:
    subjectwise_epochs = pickle.load(f)

ch_list = subjectwise_epochs[0].info["ch_names"]

for i, name in enumerate(ch_list):
    print(i, name)