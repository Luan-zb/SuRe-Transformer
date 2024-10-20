# python splits_k_fold.py
# python splits_k_fold.py --csv_path dataset_csv/cohort_for_3_quantile.csv --save_path splits/HRD_HRP_5_folds_30n_1000s_3_quantile

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse
import os
parser = argparse.ArgumentParser(description="Creating splits for HRD/HRP classification")
parser.add_argument(
    '--csv_path',
    type=str,
    default='dataset_csv/cohort_for_2_quantile.csv',
    help="path to csv file"
)
parser.add_argument(
    '--save_path',
    type=str,
    default="splits/HRD_HRP_5_folds_30n_1000s",
    help="path to save splits"
)
parser.add_argument(
    '--fold',
    type=int,
    default=5,
    help="number of folds"
)
def main(args):
    # Assuming your data is stored in a DataFrame named df
    # Replace this with your actual DataFrame
    df = pd.read_csv(args.csv_path)
    unique_case_ids = df['case_id'].unique()
    print("unique_case_ids: ", unique_case_ids)
    
    kf = KFold(n_splits=args.fold, random_state=42, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(unique_case_ids)):
    # for train_index, test_index in kf.split(unique_case_ids):
        print(f"train_index: {train_index}")
        # Split train_data into train and validation
        train_case_ids, val_case_ids = train_test_split(train_index, test_size=1/4, random_state=42)
        print(f"train_case_ids: {train_case_ids}")
        test_case_ids= test_index
        print(f"test_case_ids: {test_case_ids}")
        
        print(unique_case_ids[train_case_ids])

        train_data = df[df['case_id'].isin(unique_case_ids[train_case_ids])]
        val_data = df[df['case_id'].isin(unique_case_ids[val_case_ids])]
        test_data = df[df['case_id'].isin(unique_case_ids[test_case_ids])]

        # Display the results
        print(f"\nFold {fold + 1} - Train Set:")
        print(train_data)
        print(f"\nFold {fold + 1} - Validation Set:")
        print(val_data)
        print(f"\nFold {fold + 1} - Test Set:")
        print(test_data)

        train_slide_ids = train_data['slide_id']
        print(f"len(train_slide_ids) in Fold {fold + 1}: {len(train_slide_ids)}")
        val_slide_ids = val_data['slide_id']
        print(f"len(val_slide_ids) in Fold {fold + 1}: {len(val_slide_ids)}")
        test_slide_ids = test_data['slide_id']
        print(f"len(test_slide_ids) in Fold {fold + 1}: {len(test_slide_ids)}")

        # Create a new DataFrame for storing the slide_ids
        splits_test_df = pd.DataFrame({
            'train': train_slide_ids,
            'val': val_slide_ids,
            'test': test_slide_ids
        })

        # Create a directory if it doesn't exist
        os.makedirs(args.save_path, exist_ok=True)

        splits_test_df.to_csv(f"{args.save_path}/splits_{fold}.csv", index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


