import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
parser=argparse.ArgumentParser(description="get cohort csv")
parser.add_argument(
    "--tcga_breast_h5_dir",
    help="path of tcga breast h5 file",
    default="/work1/lhj/project/InfoTransformer/patch_clustering/kmeans_clustering_8K/features_30n_proportion/30n_1200s",
    type=str
)


def process_data(h5_Dir,slide_csv,clin_csv):
    brca_label_df = pd.DataFrame(columns=["case_id", "slide_id", "label"])
    for fname in os.listdir(h5_Dir):
        if fname.endswith('.h5'):
            temp_df = pd.DataFrame({"case_id": fname[:12], "slide_id": os.path.splitext(fname)[0], "label": "BRCA"}, index=[0])
            brca_label_df = pd.concat([temp_df, brca_label_df], ignore_index=True)
    print(brca_label_df)

    brca_info = r'./all_cancer_info.csv'
    brca_info_df = pd.read_csv(brca_info)
    brca_info_df = brca_info_df[brca_info_df['class'] == 'BRCA']
    brca_info_df = brca_info_df.dropna(subset=['HRD_Score'])
    hrd_df = brca_info_df[['patient_barcode', 'TCGA sample barcode', 'class', 'subtype', 'HRD_TAI', 'HRD_LST', 'HRD_LOH', 'HRD_Score']]
    
    # 合并case_id.csv和HRD_score.csv
    case_id_df = brca_label_df
    merged_df = pd.merge(case_id_df, hrd_df, left_on='case_id', right_on='patient_barcode', how='inner')
    result_df = merged_df[['patient_barcode', 'TCGA sample barcode', 'slide_id', 'class', 'subtype', 'HRD_TAI', 'HRD_LST', 'HRD_LOH', 'HRD_Score']]

    # 获取HRD/HRP标签
    hrd_hrp_label_df = result_df
    median_hrd_score = hrd_hrp_label_df['HRD_Score'].median()
    hrd_hrp_label_df['label'] = hrd_hrp_label_df['HRD_Score'].apply(lambda x: 'HRD' if x > median_hrd_score else 'HRP')

    # 合并性别和年龄
    clin_info_csv = "./brca_clin_info.csv"
    clin_info_df = pd.read_csv(clin_info_csv)
    merged_df = pd.merge(hrd_hrp_label_df, clin_info_df, left_on='patient_barcode', right_on='Patient ID', how='inner')
    df = merged_df[['patient_barcode', 'TCGA sample barcode', 'slide_id', 'class', 'subtype', 'HRD_TAI', 'HRD_LST', 'HRD_LOH', 'HRD_Score', 'label', 'Sex', 'Diagnosis Age']]

    slide_df = df[['patient_barcode', 'slide_id']]
    slide_df = slide_df.rename(columns={'patient_barcode': 'PATIENT', 'slide_id': 'FILENAME'})
    slide_df.to_csv(slide_csv, index=False)
    
    clin_df = df[['patient_barcode', 'label', 'Sex', 'Diagnosis Age']]
    clin_df = clin_df.rename(columns={'patient_barcode': 'PATIENT', 'label': 'TARGET', 'Sex': 'Gender', 'Diagnosis Age': 'AGE'})
    clin_df = clin_df.drop_duplicates(subset='PATIENT')
    clin_df.to_csv(clin_csv, index=False)


if __name__=='__main__':
    # from the h5.file to generate the csv file(case_id,slide_id,label)
    args=parser.parse_args()
    h5_Dir = args.tcga_breast_h5_dir
    clin_csv="clinic.csv"
    slide_csv="brca_slide.csv"
    process_data(h5_Dir,slide_csv,clin_csv)
    
