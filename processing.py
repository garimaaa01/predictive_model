# processing.py --1
import mne
import os
import pickle


def main():

    data_folder=os.path.join(".","eeg_data","derivatives")
        
    rename_dict = {
            "T3": "T7",
            "T4": "T8",
            "T5": "P7",
            "T6": "P8"
        }

    all_raw=[]
    subjectwise_epochs=[]

    for subjects in sorted(os.listdir(data_folder)):
        eeg_file=os.path.join(data_folder,subjects,"eeg")
        if not os.path.isdir(eeg_file):
            continue
        for f in sorted(os.listdir(eeg_file)):
            if f.endswith(".set"):
                file_path=os.path.join(eeg_file,f)
                raw=mne.io.read_raw_eeglab(file_path,preload=True)
                    
                    #--------------
                    # Preprocessing
                    #--------------
                raw.rename_channels(rename_dict)
                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage, match_case=False, on_missing="ignore")
                raw.set_eeg_reference('average', projection=False)
                all_raw.append(raw)
        
                    #--------------
                    # Epoch 
                    #---------------
                raw.crop(tmin=30.0,tmax=300)
                epochs= mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.0, preload=True)
                subjectwise_epochs.append(epochs)



  
    with open("all_epochs.pkl", "wb") as f:
        pickle.dump(subjectwise_epochs, f)




if __name__=="__main__":
    main()