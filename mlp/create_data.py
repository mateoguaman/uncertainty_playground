# Create Dataset

import os
import numpy as np
import matplotlib.pyplot as plt

def create_dataset(save_dir):
    '''Create dataset for function (2x-4) * sin(x) with x \in [-20, 20].

    Args:
        - save_dir:
            String, directory where dataset will be saved. Dataset will be saved in two separate files: inputs.npy, labels.npy
    '''

    # Generate data
    data_size = 100
    x_balanced = np.random.uniform(-20, 20, data_size)
    y_balanced = (2*x_balanced - 4) * np.sin(x_balanced)

    x_imbalanced = np.concatenate([np.random.uniform(-5,5,int(0.8*data_size)), 
                            np.random.uniform(-20,-5,int(0.1*data_size)), 
                            np.random.uniform(5,20,int(0.1*data_size))], axis=0)
    y_imbalanced = (2*x_imbalanced - 4) * np.sin(x_imbalanced)

    x_gt = np.linspace(-20, 20, 1000)
    y_gt = (2*x_gt - 4) * np.sin(x_gt)

    plt.plot(x_gt, y_gt, c='red', label="GT Function")
    plt.scatter(x_balanced, y_balanced, c='blue', alpha=0.5, label="Balanced Dataset")
    plt.scatter(x_imbalanced, y_imbalanced, c='green', alpha=0.5, label="Imbalanced Dataset")
    plt.legend()
    plt.grid()
    plt.title("Toy Dataset: (2x-4)*sin(x)")
    plt.show()

    # import pdb;pdb.set_trace()
    gt_dir = os.path.join(save_dir, "ground_truth")
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    gt_inputs_dir = os.path.join(gt_dir, "inputs.npy")
    gt_labels_dir = os.path.join(gt_dir, "labels.npy")
    np.save(gt_inputs_dir, x_gt)
    np.save(gt_labels_dir, y_gt)

    balanced_dir = os.path.join(save_dir, "balanced")
    if not os.path.exists(balanced_dir):
        os.makedirs(balanced_dir)
    balanced_inputs_dir = os.path.join(balanced_dir, "inputs.npy")
    balanced_labels_dir = os.path.join(balanced_dir, "labels.npy")
    np.save(balanced_inputs_dir, x_balanced)
    np.save(balanced_labels_dir, y_balanced)

    imbalanced_dir = os.path.join(save_dir, "imbalanced")
    if not os.path.exists(imbalanced_dir):
        os.makedirs(imbalanced_dir)
    imbalanced_inputs_dir = os.path.join(imbalanced_dir, "inputs.npy")
    imbalanced_labels_dir = os.path.join(imbalanced_dir, "labels.npy")
    np.save(imbalanced_inputs_dir, x_imbalanced)
    np.save(imbalanced_labels_dir, y_imbalanced)


if __name__ == "__main__":
    save_dir = "/home/micah/airlab/uncertainty_playground/mlp/data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    create_dataset(save_dir)