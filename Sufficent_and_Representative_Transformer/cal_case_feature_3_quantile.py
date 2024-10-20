import os
import h5py
import csv
import pandas as pd
import argparse

parser=argparse.ArgumentParser(description="get cohort csv")
parser.add_argument(
    "--tcga_breast_h5_dir",
    help="path of tcga breast h5 file",
    default="/work1/lhj/project/InfoTransformer/patch_clustering/kmeans_clustering_8K/features_30n_proportion/30n_1200s",
    type=str
)
parser.add_argument(
    "--save_cohort_h5_csv",
    help="path of save cohort h5 csv, (case_id,slide_id)",
    default="dataset_csv/cohort_for_3_quantile.csv",
    type=str
)
def generate_from_all_h5(rDir):
    result = []
    for root, dirs, files in os.walk(rDir):
        for file_name in files:
            if file_name.endswith('.h5'):
                h5_file_path = os.path.join(root, file_name)
                result.append([file_name[:12], os.path.splitext(file_name)[0]])
    return result

def get_cohort_csv(case_slide_csv, hrd_hrp_label_csv, save_cohort_csv):
    label_csv = pd.read_csv(hrd_hrp_label_csv)
    merged_df = pd.merge(case_slide_csv, label_csv, left_on='case_id', right_on='patient_barcode', how='inner')
    merged_df[['case_id', 'slide_id', 'label']].to_csv(save_cohort_csv, index=False)
    print(merged_df['case_id'].shape[0]) 
    print(merged_df['case_id'].unique().shape[0])     

if __name__ == '__main__':
    args = parser.parse_args()
    tcga_breast_h5_dir = args.tcga_breast_h5_dir
    case_slide_data = generate_from_all_h5(tcga_breast_h5_dir)
    case_slide_csv = pd.DataFrame(case_slide_data, columns=["case_id", "slide_id"])
    print(case_slide_csv['case_id'].shape[0]) 
    print(case_slide_csv['case_id'].unique().shape[0])  
    hrd_hrp_label_csv = "HRD_HRP_for_case/HRD_label_3_quantile.csv"
    save_cohort_csv = args.save_cohort_h5_csv
    get_cohort_csv(case_slide_csv, hrd_hrp_label_csv, save_cohort_csv)