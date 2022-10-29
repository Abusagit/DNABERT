import numpy as np

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import argparse
import pandas as pd
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_df", type=str)
    parser.add_argument("--kmer", default=6, type=int)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--input_delim", default=',')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default=".")
    
    
    return parser

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def main():
    
    parser = get_parser()
    args = parser.parse_args()
    
    processed_data = []
    
    input_df = pd.read_csv(args.input_df, sep=args.input_delim)
    
    for _, (sequence, label) in tqdm(input_df.iterrows(), total=input_df.shape[0]):
        kmer_tokens = seq2kmer(sequence, args.kmer)
        processed_data.append([kmer_tokens, label])
    
    
    processed_data = np.array(processed_data)
    
    train_data, test_data = train_test_split(processed_data, test_size=args.test_ratio, shuffle=True, random_state=args.seed)
    
    
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    train_df = pd.DataFrame(train_data, columns=["sequence", "label"])
    train_df.to_csv(outdir / "train.tsv", sep="\t", index=False)
    
    test_df = pd.DataFrame(test_data, columns=["sequence", "label"])
    test_df.to_csv(outdir / "dev.tsv", sep="\t", index=False)
    

if __name__ == '__main__':
    main()
    