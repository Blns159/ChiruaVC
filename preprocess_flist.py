import os
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    train_list = "./filelists/train.txt"
    val_list = "./filelists/val.txt"
    test_list = "./filelists/test.txt"
    source_dir = "./dataset/audio_vlsp"
    train = []
    val = []
    test = []
    idx = 0
    
    for speaker in tqdm(os.listdir(source_dir)):
        wavs = os.listdir(os.path.join(source_dir, speaker))
        shuffle(wavs)
        train += wavs[2:-10]
        val += wavs[:2]
        test += wavs[-10:]
        
    shuffle(train)
    shuffle(val)
    shuffle(test)
            
    print("Writing", train_list)
    with open(train_list, "w") as f:
        for fname in tqdm(train):
            speaker = fname.split("_")[0] + "_" + fname.split("_")[1]
            wavpath = os.path.join(source_dir, speaker, fname)
            f.write(wavpath + "\n")
        
    print("Writing", val_list)
    with open(val_list, "w") as f:
        for fname in tqdm(val):
            speaker = fname.split("_")[0] + "_" + fname.split("_")[1]
            wavpath = os.path.join(source_dir, speaker, fname)
            f.write(wavpath + "\n")
            
    print("Writing", test_list)
    with open(test_list, "w") as f:
        for fname in tqdm(test):
            speaker = fname.split("_")[0] + "_" + fname.split("_")[1]
            wavpath = os.path.join(source_dir, speaker, fname)
            f.write(wavpath + "\n")
            