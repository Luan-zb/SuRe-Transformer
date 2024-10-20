import numpy as np
import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
import h5py
import glob
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pathlib import Path

parser = argparse.ArgumentParser(description='K-menas for json file')
parser.add_argument('--class_num', default=50, metavar='CLASS_NUM', type=int, help='Clustering Number Class')
parser.add_argument('--method', default='kmeans', type=str, help='Methods to clustering')
parser.add_argument('--feat_path',default="/data/Project/luanhaijing/MSI-MIL/TCGA-DINO/features/breast_20X_breast_8K/h5_files/256px_dino_2mpp_0xdown_normal",
                    metavar='WSI_PATH', type=str,help='Path to the input WSI file')
parser.add_argument('--png_path', default='./kmeans_clustering_8K/brca_all_dino_feat_clustering', metavar='PNG_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--txt_path', default='./kmeans_clustering_8K/brca_all_dino_feat_clustering', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')

random = np.random.RandomState(0)
def randomcolor(class_num):
    colorList = []
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    for num in range(class_num):
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,14)]
        colorList.append("#"+color)
    return colorList

def get_colors(num_colors):
    cmap = plt.get_cmap('tab20',num_colors)
    colors = [cmap(i) for i in range(num_colors)]
    return colors

# K-Means聚类
def kmeans_train(features, clusters):
    return KMeans(n_clusters=clusters).fit(features)

# 谱聚类
def spectral_train(features, clusters):
    return SpectralClustering(n_clusters=clusters).fit(features)

# 模糊C均值聚类
def fcm_train(features, clusters):
    feature_t = np.array(features).T
    center, u, u0, d, jm, p, fpc = cmeans(feature_t, m=2, c=clusters, error=0.0001, maxiter=1000)
    return np.argmax(u, axis=0)  

def pca_train(features, cls_num):
    pca = PCA(n_components=cls_num,svd_solver="arpack")
    new_feature = pca.fit_transform(features)
    return new_feature


def showEvaluate(args,feat_indices, feat_name, features, labels, json_name, ext):
    data = (list(zip(feat_indices, feat_name, labels)))
    data = pd.DataFrame(data)
    json_name = json_name.replace('.h5', ext + '_cls.txt')
    
    txt_path=Path(args.txt_path)
    txt_path.mkdir(parents=True, exist_ok=True)
    txt_path = os.path.join(txt_path, json_name)
    data.to_csv(txt_path, sep='\t', index=0, header=0)
  

def showClass(args, json_name, feat_name, features, labels, color_list, ext=''):
    png_path=Path(args.png_path)
    png_path.mkdir(parents=True, exist_ok=True)
    png_file = os.path.join(png_path, json_name.replace('.h5', ext + '.png'))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    coords=[]

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for i, coord in enumerate(feat_name):
        x, y = int(coord[1].item()), int(coord[2].item())  
        x_min, x_max = min(x, x_min), max(x, x_max)
        y_min, y_max = min(y, y_min), max(y, y_max)
        coords.append(coord)
        plt.gca().add_patch(plt.Rectangle((int(coord[1]), int(coord[2])), 256, 256, linewidth=1, edgecolor=color_list[labels[i]], facecolor=color_list[labels[i]]))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_min - 5000, x_max + 5000)
    plt.ylim(y_min - 5000, y_max + 5000)
    plt.savefig(png_file, bbox_inches='tight', dpi=1000)
    plt.clf()
    

def loadDataSet(h5_path):
    h5_file = h5py.File(h5_path)
    if 'feats' in h5_file.keys():   
        features = torch.Tensor(np.array(h5_file['feats']))
        feat_name = torch.Tensor(np.array(h5_file['coords'])) 
        num_features = features.shape[0]
        feat_indices = np.arange(num_features, dtype=int)
        return feat_indices,feat_name, features
    else:
        raise ValueError("Required dataset 'feats' not found in the file")
def run(args):
    color_list = get_colors(args.class_num)
    paths = glob.glob(os.path.join(args.feat_path, '*.h5'))
    paths.sort()
    
    silhouette_scores = []
    calinski_harabasz_scores = []
    for path in paths:
        json_name=os.path.basename(path)
        if os.path.exists(os.path.join(args.png_path, json_name.replace('.h5', 'kmeans.png'))):
            continue
    
        feat_indices, feat_name, features = loadDataSet(path)

        cls_nums = args.class_num if len(features)>=args.class_num else len(features)
        
        print('Clustering....')
        if args.method == 'kmeans':
            print('Running K-Means clustering...')
            kmeans_cls = kmeans_train(features, cls_nums)
        elif args.method == 'spectral':
            print('Running Spectral clustering...')
            kmeans_cls= spectral_train(features, cls_nums)
        elif args.method == 'fcm':
            print('Running Fuzzy C-Means clustering...')
            kmeans_cls= fcm_train(features, cls_nums)
        elif args.method == 'pca':
            print('Running PCA dimension reduction...')
            kmeans_cls= pca_train(features, cls_nums)
            
            
        print("calculating the SH score....")
        silhouette_avg = silhouette_score(features, kmeans_cls.labels_)
        print(f"The average silhouette_score is : {silhouette_avg}")
        
        print("calculating the CH score....")
        calinski_harabasz_index = calinski_harabasz_score(features, kmeans_cls.labels_)
        print(f"The Calinski-Harabasz Index is : {calinski_harabasz_index}")
        
        silhouette_scores.append(silhouette_avg)
        calinski_harabasz_scores.append(calinski_harabasz_index)
        
        print('Writing...')
        showEvaluate(args, feat_indices, feat_name, features, kmeans_cls.labels_, json_name, args.method)
        print('Ploting...')
        showClass(args, json_name, feat_name, features, kmeans_cls.labels_,color_list, args.method)
        
    avg_silhouette_score = np.mean(silhouette_scores)
    avg_calinski_harabasz_score = np.mean(calinski_harabasz_scores)
    print(f"Average Silhouette Score: {avg_silhouette_score}")
    print(f"Average Calinski-Harabasz Score: {avg_calinski_harabasz_score}")
def main():
    args= parser.parse_args()
    run(args)
    
if __name__=='__main__':
    main()