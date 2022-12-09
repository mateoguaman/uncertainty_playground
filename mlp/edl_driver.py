import functools
import numpy as np
import matplotlib.pyplot as plt

import evidential_deep_learning as edl
import sys

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as d
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wandb
from collections import OrderedDict
import imageio.v2 as imageio

# Upload test data to WandB
USE_WANDB = True
DEBUG = True

def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics

# Load Toy Dataset from generated data (create_data.py)
class ToyDataset(Dataset):
    # Constructor
    def __init__(self, inputs_fp, labels_fp):
        self.inputs = np.load(inputs_fp)
        self.labels = np.load(labels_fp)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class Network(nn.Module):
    # Constructor
    # This network may be defined incorrectly
    def __init__(self, mlp_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=1, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            # nn.ReLU(),
            # nn.Linear(in_features=mlp_size, out_features=2),

            # Make evidential distribution         
            edl.layers.DenseNormalGamma(mlp_size, 1),
        )
    
    # Forward pass through network
    def forward(self, input_data):
        out = self.model(input_data)

        return out

def EvidentialRegressionLoss(true, pred):
    return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)

def run_train_epoch(model, train_dataloader, optimizer, epoch):
    model.train()

    # Initiate metrics, inputs, labels lists
    all_metrics = []
    all_inputs = []
    all_labels = []

    # criterion = nn.MSELoss(reduction="sum")

    for i, (input, label) in enumerate(train_dataloader):
        input = input.view(-1, 1).float()
        label = label.view(-1, 1).float()
        input = input.cuda() # "use Cuda"
        label = label.cuda()
        output = model(input)

        # eps = torch.randn(output.shape[0]).cuda()
        # prediction = (output[:,0] + torch.exp(output[:,1] / 2) * eps).unsqueeze(1)

        all_inputs.append(input.squeeze().detach().cpu())
        all_labels.append(label.squeeze().detach().cpu())

        # loss = criterion(prediction, label)/prediction.shape[0]
        # gaussian = d.normal.Normal(output[:,0], torch.exp(output[:,1]/2))
        # loss = -torch.sum(gaussian.log_prob(label.squeeze()))/label.shape[0]

        loss = EvidentialRegressionLoss(label, output) # no need to transform b/c softplus already

        all_metrics.append(OrderedDict(loss=loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # wandb.log(data={"train/loss": loss, "train/epoch": epoch})

    all_inputs = torch.cat(all_inputs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return avg_dict(all_metrics), all_inputs, all_labels

def get_test_metrics(model, test_dataloader, epoch):
    model.eval()
    all_metrics = []
    # criterion = nn.MSELoss(reduction="sum")

    all_inputs = []
    all_means = []
    all_stds  = []
    all_labels = []

    with torch.no_grad():
        for i, (input, label) in enumerate(test_dataloader):
            input = input.cuda()
            label = label.cuda()
            input = input.view(-1, 1).float()
            label = label.view(-1, 1).float()
            output = model(input)

            # eps = torch.randn(output.shape[0]).cuda()
            # prediction = (output[:,0] + torch.exp(output[:,1] / 2) * eps).unsqueeze(1)
            # gaussian = d.normal.Normal(output[:,0], torch.exp(output[:,1]/2))
            # loss = -torch.sum(gaussian.log_prob(label.squeeze()))/label.shape[0]

            loss = EvidentialRegressionLoss(label, output)

            all_inputs.append(input.squeeze().detach().cpu())
            all_means.append(output[:,0].squeeze().detach().cpu())
            all_stds.append(torch.exp(output[:,1]/2).squeeze().detach().cpu())
            all_labels.append(label.squeeze().detach().cpu())

            # loss = criterion(prediction, label)/prediction.shape[0] # TODO logprob with torch distributions

            all_metrics.append(OrderedDict(loss=loss))

            # wandb.log(data={"test/loss": loss, "test/epoch": epoch})

    all_inputs = torch.cat(all_inputs).numpy()
    all_means = torch.cat(all_means).numpy()
    all_stds = torch.cat(all_stds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return avg_dict(all_metrics), all_inputs, all_means, all_stds, all_labels

# def get_plots(model, test_dir):

def main(train_dir, test_dir, batch_size, lr, num_epochs, mlp_size=32, save_img_interval=10, run=0):

    train_inputs_fp = os.path.join(train_dir, "inputs.npy")
    test_inputs_fp = os.path.join(test_dir, "inputs.npy")
    train_labels_fp = os.path.join(train_dir, "labels.npy")
    test_labels_fp = os.path.join(test_dir, "labels.npy")

    train_dataset = ToyDataset(train_inputs_fp, train_labels_fp)
    test_dataset  = ToyDataset(test_inputs_fp, test_labels_fp)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Network(mlp_size=mlp_size)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr) # EDL paper uses 5e-4

    if USE_WANDB:
        config = {
            'mlp_size': mlp_size,
            'lr': lr
        }
        wandb.init(project="MLP", group="edl", name=f"imbalanced_edl_take2_mlp{mlp_size}_lr{lr}_run{run}", reinit=True, config=config)

    image_filenames = []
    images_dir = os.path.join("/home/micah/airlab/uncertainty_playground/mlp/viz/img", f"imbalanced_edl_take2_mlp{mlp_size}_lr{lr}_run{run}")
    gif_dir = os.path.join("/home/micah/airlab/uncertainty_playground/mlp/viz", f"imbalanced_edl_take2_mlp{mlp_size}_lr{lr}_run{run}")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    for epoch in range(num_epochs):
        train_metrics, all_inputs_train, all_labels_train = run_train_epoch(model, train_dataloader, optimizer, epoch)
        test_metrics, all_inputs, all_means, all_stds, all_labels = get_test_metrics(model, test_dataloader, epoch)

        print(f"Epoch: {epoch}.")
        print(f"Train metrics:")
        print(train_metrics)
        print(f"Test metrics: ")
        print(test_metrics)

        if epoch % save_img_interval == 0:
            plt.plot(np.sort(all_inputs), all_labels[np.argsort(all_inputs)], c='red', label="GT Function")
            # plt.scatter(all_inputs, all_outputs, c='blue', alpha=0.5, label="Predicted function")
            # plt.errorbar(all_inputs, all_means, yerr=all_stds, alpha=0.5, fmt='o')
            plt.plot(np.sort(all_inputs), all_means[np.argsort(all_inputs)], c='blue', alpha=0.5, label="Predicted function")
            plt.fill_between(np.sort(all_inputs), all_means[np.argsort(all_inputs)] - all_stds[np.argsort(all_inputs)], all_means[np.argsort(all_inputs)] + all_stds[np.argsort(all_inputs)], color="blue", alpha=0.3)
            plt.scatter(all_inputs_train, all_labels_train, c="green", alpha=0.5, label="Training data")
            plt.legend(loc='lower center')
            plt.grid()
            plt.ylim([-50.0, 50.0])
            plt.title(f"Imbalanced Toy Dataset: (2x-4)*sin(x). Epoch {epoch}/{num_epochs}.")
            # plt.show()

            filename = os.path.join(images_dir, f"{epoch}.png")
            image_filenames.append(filename)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

        train_metrics = {"train/"+k:v for k,v in train_metrics.items()}
        test_metrics = {"test/"+k:v for k,v in test_metrics.items()}
        if USE_WANDB:
            wandb.log(data=train_metrics, step=epoch)
            wandb.log(data=test_metrics, step=epoch)
        # wandb.log({"test/plot": wandb.Image(plt)})
        
    # build gif
    gif_path = os.path.join(gif_dir, f"imbalanced_edl_take2_mlp{mlp_size}_lr{lr}_run{run}.gif") 
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    if USE_WANDB:
        wandb.log({"test/video": wandb.Video(gif_path, fps=4, format="gif")})
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)
    wandb.finish()


if __name__=="__main__":
    train_dir = "/home/micah/airlab/uncertainty_playground/mlp/data/imbalanced"
    test_dir = "/home/micah/airlab/uncertainty_playground/mlp/data/ground_truth"

    batch_size = 100
    lr = 7e-4
    num_epochs = 5000
    mlp_size = 32
    save_img_interval = 50
    # run = 4

    # main(train_dir, test_dir, batch_size, lr, num_epochs, mlp_size, save_img_interval, run)

    for run in range(5):
        main(train_dir, test_dir, batch_size, lr, num_epochs, mlp_size, save_img_interval, run)