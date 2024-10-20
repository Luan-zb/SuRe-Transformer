import os
import argparse
import yaml
import h5py
import random
import torch
parser = argparse.ArgumentParser(description='get Features for histopathology images argumentation')
parser.add_argument(
    "--h5_file_path",
    help="file to save the h5 ",
    default="/data/Project/luanhaijing/MSI-MIL/TCGA-DINO/features/breast_20X_breast_8K/h5_files/256px_dino_2mpp_0xdown_normal",
    type=str
)
parser.add_argument(
    "--output_h5_file_path",
    help="Directory for storing expanded h5 files",
    default="/data/Project/luanhaijing/MSI-MIL/InfoTransformer/2_get_feature_multi/features_argumented_8K",
    type=str
)
parser.add_argument(
    "--num_samples",
    help="number of samples",
    default=30,
    type=int
)
parser.add_argument(
    "--sample_size",
    help="size of sample",
    default=800,
    type=int
)
def save_hdf5(sample, sample_dir):
    with h5py.File(sample_dir,"w",) as f:
        f["feats_index"] = sample

def main(args):
    h5_fileDir=args.h5_file_path
    h5_files=os.listdir(h5_fileDir)
    unprocessed_dir=[]
    paths=[]
    features=[]
    count=0
    random.seed(42)
    for h5_file in h5_files:
        if h5_file.split(".")[-1] == "h5":
            wsi_name=os.path.splitext(h5_file)[0]
            h5_path=os.path.join(h5_fileDir,h5_file)
            paths.append(h5_path)
            
            output_name = f"{args.output_h5_file_path}/{args.num_samples}n_{args.sample_size}s"
            feature=h5py.File(h5_path,'r')['feats']
            features.append(len(feature))
            if len(feature) <=3000:
                unprocessed_dir.append(h5_path)
                count+=1
                continue
            else:
                num_samples=args.num_samples
                sample_size=args.sample_size
                with h5py.File(h5_path) as h5f:
                    feature=h5f['feats']
                    num_features=len(feature)
                    if num_features > sample_size:
                        for i in range(1,num_samples+1):
                            sample=random.sample(range(num_features), sample_size)
                            sample=sorted(sample)
                            print(sample)
                            sample_name=f"{wsi_name}_{i}.h5"
                            sample_dir=os.path.join(output_name, sample_name) 
                            os.makedirs(os.path.dirname(sample_dir), exist_ok=True)
                            save_hdf5(sample, sample_dir)
                    else:
                        for i in range(1,num_samples+1):
                            sample = random.choices(range(num_features), k=sample_size)
                            sample=sorted(sample)
                            print(sample)
                            sample_name=f"{wsi_name}_{i}.h5"
                            sample_dir=os.path.join(output_name, sample_name) 
                            os.makedirs(os.path.dirname(sample_dir), exist_ok=True)
                            save_hdf5(sample, sample_dir)         
    print("unprocessed_dir:",unprocessed_dir)            
    print("features:",features)  
    print("count:",count)
        
if __name__ == '__main__':
    args=parser.parse_args()
    main(args)