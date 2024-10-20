import pandas as pd
import os
import csv
import numpy as np

# get the tcga brca dataset's HRD/HRP label
def get_HRD_label_for_3_quantile(all_info,hrd_hrp_csv):
    # from the all_cancer_info.csv to get the HRD score
    #(patient_barcode,TCGA sample barcode,class subtype,HRD_TAI,HRD_LST,HRD_LOH,HRD_Score)
    brca_info=pd.read_csv(all_info)
    print(brca_info)

    df=pd.DataFrame(brca_info)
    # get the information of the brca patients
    df = df[df['class'] == 'BRCA']
    df = df.dropna(subset=['HRD_Score'])
    print(df)

    # get the information of the brca patients with HRD score
    df= df[['patient_barcode', 'TCGA sample barcode', 'class', 'subtype', 'HRD_TAI', 'HRD_LST', 'HRD_LOH', 'HRD_Score']]
    
    hrd_scores = df['HRD_Score'].to_numpy()

    # 计算上下三分位数
    q1 = np.percentile(hrd_scores , 33)
    print("q1", q1)
    q2 = np.percentile(hrd_scores, 66)
    print("q2", q2)

    # 划分标签
    # labels = np.where(hrd_scores <= q1, 'HRP', np.where(hrd_scores >= q2, 'HRD', 'Uncertain'))
    # print(labels)

    df['label']  = np.where(df['HRD_Score'] <= q1, 'HRP', np.where(df['HRD_Score'] >= q2, 'HRD', 'Uncertain'))
    filtered_df = df[(df['label'] == 'HRP') | (df['label'] == 'HRD')]
    print(filtered_df['patient_barcode'].nunique())     
    filtered_df[['patient_barcode', 'label']].to_csv(hrd_hrp_csv, index=False)



if __name__=='__main__':
    all_info=r'all_cancer_info.csv'
    HRD_label=r'HRD_label_3_quantile.csv'  
    get_HRD_label_for_3_quantile(all_info,HRD_label)
    