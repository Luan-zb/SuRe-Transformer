import pandas as pd
import os
import csv
# get the tcga brca dataset's HRD/HRP label
def get_HRD_label_for_2_quantile(all_info,hrd_hrp_csv):
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
    
    # change the HRD score to the label(HRD or HRP) in 2_quantify
    median_hrd_score = df['HRD_Score'].median()
    print("median_hrd_score", median_hrd_score)

    df['label'] = df['HRD_Score'].apply(lambda x: 'HRD' if x > median_hrd_score else 'HRP')
    print(df['patient_barcode'].nunique())      
    df[['patient_barcode', 'label']].to_csv(hrd_hrp_csv, index=False)

if __name__=='__main__':  
    all_info=r'all_cancer_info.csv'
    HRD_label=r'HRD_label_2_quantile.csv'  
    get_HRD_label_for_2_quantile(all_info,HRD_label)
    