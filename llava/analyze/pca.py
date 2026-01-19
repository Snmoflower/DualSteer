import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def perform_pca_analysis(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith(".npy"):
            layer_num = filename.split('_')[-1].split('.')[0]
            output = np.load(os.path.join(output_dir, filename))
            
            # 将三维数组展平成二维数组
            _, seq_len, hidden_size = output.shape
            output_flat = output.reshape(seq_len, hidden_size)
            
            # 检查 n_samples 和 n_features 是否大于等于 2
            if output_flat.shape[0] >= 2 and output_flat.shape[1] >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(output_flat)
                plt.figure()
                plt.scatter(pca_result[:, 0], pca_result[:, 1])
                plt.title(f'PCA of Self Attention Layer {layer_num}')
                plt.savefig(os.path.join(output_dir, f'pca_layer_{layer_num}.png'))
                plt.close()
            else:
                print(f"Skipping PCA for layer {layer_num} due to insufficient samples or features.")

if __name__ == "__main__":
    output_dir = "cross_attention_outputs"
    perform_pca_analysis(output_dir)