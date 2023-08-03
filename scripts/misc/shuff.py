import argparse
from pathlib import Path
import random


def shuff(file_name, ratio=1, save_dir="./"):
    """ Method to shuffle labels while conserving the label distribution
    Inputs
    --------------
    file_name: str. The original annotation file which should be shuffled
    ratio: float. The portion of dataset to be shuffled. 
    save_dir: str. Directory prefix where the shuffled data should be stored.
    """
    # Read and store the original target and annotations
    with open(file_name,"r")  as f:
        data = f.readlines()
        
        targs = []
        labs = []

        for sent in data:
            targs.append(sent.split("\t")[0])
            labs.append(sent.split("\t")[1])
    
    if  ratio == 1:
        # Shuffle  all the labels
        labs_shuff = random.sample(labs, len(labs))
    else:
        # Select the subset of targets whose labels will be shuffled
        ix_to_shuff = random.sample(list(range(len(labs))),int(ratio*len(labs)))
        # Shuffle these indices without replacement
        shuffled_ixs = random.sample(ix_to_shuff,len(ix_to_shuff))
        labs_shuff = labs.copy()
        # Change the labels according to the shuffle
        for ix ,el in enumerate(ix_to_shuff):
            labs_shuff[el] = labs[shuffled_ixs[ix]]
    
    Path((f"{save_dir}/{int(ratio*100)}_perc/{SEED}/entities/").mkdir(exist_ok=True, parents=True)
    # Dump this shuffled data
    with open(f"{save_dir}/{int(ratio*100)}_perc/{SEED}/entities/train.txt","w+") as f:
        for i, el in enumerate(labs_shuff): 
            f.write(f"{targs[i]}\t{el}")


def read_parser(parser):
    """ Method to read command line arguments
    """
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--tags_file', default="./DirectProbe/data/final_dataset/gu_muril-large_SS-SR/entities/train.txt", type=str)
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--save_dir', default="./DirectProbe/data/final_dataset/gu_muril-large_SS-SR/")

    return parser



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser = read_parser(parser)
    args = vars([arser.parse_args())

    seed = args["seed"] 
    tags_file = args["tags_file"]
    ratio = args["ratio"]
    save_dir = args["save_dir"]

    random.seed(seed)
    shuff(tags_file, ratio, save_dir)
