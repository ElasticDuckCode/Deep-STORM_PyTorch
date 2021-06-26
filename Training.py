import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from scipy.io import savemat
from os.path import abspath
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from CNN_Model import project_01, normalize_im, DeepSTORM, l1l2loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize weights using orthogonal method seen in original tensorflow implementation
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        #nn.init.zeros_(m.bias)
    return

class SimpleSTORMDataset(Dataset):
    def __init__(self, trainX, trainY, transform=None, target_transform=None):
        self.img_labels = trainY
        self.img_points = trainX
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        return (self.img_points[idx], self.img_labels[idx])

def train_model(filename, weights_name, meanstd_name):
    # for reproducibility
    np.random.seed(123)

    # load the datasets
    matfile = h5py.File(filename, 'r')
    patches = np.array(matfile['patches'])
    heatmaps = 100.0 * np.array(matfile['heatmaps'])

    # split dataset into training and validation sets
    trainX, testX, trainY, testY = train_test_split(patches, heatmaps,\
            test_size=0.3, random_state=42)
    print(f"Number of Training Examples: {trainX.shape[0]}")
    print(f"Number of Validation Examples: {testX.shape[0]}")

    # ensure all data is float32
    trainX = trainX.astype('float32')
    trainY = trainY.astype('float32')
    testX = testX.astype('float32')
    testY = testY.astype('float32')

    ## Training Set Normalization ##
    mean_train = np.zeros(trainX.shape[0], dtype=np.float32)
    std_train = np.zeros(trainX.shape[0], dtype=np.float32)
    for i in range(trainX.shape[0]):
        trainX[i, ...] = project_01(trainX[i, ...])
        mean_train[i] = trainX[i, ...].mean()
        std_train[i] = trainX[i, ...].std()
    mean_val_train = mean_train.mean()
    std_val_train = std_train.mean()
    trainX_norm = np.zeros(trainX.shape, dtype=np.float32)
    for i in range(trainX.shape[0]):
        trainX_norm[i, ...] = normalize_im(trainX[i, ...], \
                mean_val_train, std_val_train)
    # patch size
    psize = trainX_norm.shape[1]
    trainX_norm = trainX_norm.reshape(trainX.shape[0], 1, psize, psize)

    ## Test Set Normalization ##
    mean_test = np.zeros(testX.shape[0], dtype=np.float32)
    std_test = np.zeros(testX.shape[0], dtype=np.float32)
    for i in range(testX.shape[0]):
        testX[i, ...] = project_01(testX[i, ...])
        mean_test[i] = testX[i, ...].mean()
        std_test[i] = testX[i, ...].std()
    mean_val_test = mean_test.mean()
    std_val_test = std_test.mean()
    testX_norm = np.zeros(testX.shape, dtype=np.float32)
    for i in range(testX.shape[0]):
        testX_norm[i, ...] = normalize_im(testX[i, ...], \
                mean_val_test, std_val_test)
    testX_norm = testX_norm.reshape(testX.shape[0], 1, psize, psize)

    # save mean and std to mat file
    mdict = {"mean_test": mean_val_test, "std_test": std_val_test}
    savemat(meanstd_name, mdict)

    # label reshaping
    trainY = trainY.reshape(trainY.shape[0], 1, psize, psize)
    testY = testY.reshape(testY.shape[0], 1, psize, psize)

    # put data into PyTorch dataset
    train_dataset = SimpleSTORMDataset(trainX_norm, trainY)
    train_loader = DataLoader(train_dataset, \
            batch_size=16, shuffle=True, num_workers=4)
    test_dataset = SimpleSTORMDataset(testX_norm, testY)
    test_loader = DataLoader(test_dataset, \
            batch_size=16, shuffle=True, num_workers=4)

    # create Deep-STORM model
    model = DeepSTORM((psize, psize)).to(device)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
            factor=0.1, patience=5, min_lr=0.00005)
    criterion = l1l2loss


    # begin training the model
    num_epochs = 100
    train_loss_hist = []
    val_loss_hist = []
    min_val_loss = np.Inf
    for epoch in range(num_epochs):
        loop1 = tqdm(train_loader, total=len(train_loader), leave=False)
        loop2 = tqdm(test_loader, total=len(test_loader), leave=False)

        train_loss = 0
        model.train()
        for data, label in loop1:
            data, label = data.to(device), label.to(device)
            spikes_pred = model(data)
            loss = criterion(label, spikes_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item() * data.size(0)
            train_loss_hist.append(train_loss)

            loop1.set_description(f"Epoch [{epoch+1}/{num_epochs}] Training")
            loop1.set_postfix(loss = loss.item())

        val_loss = 0
        model.eval()
        for data, label in loop2:
            data, label = data.to(device), label.to(device)
            spikes_pred = model(data)
            loss = criterion(label, spikes_pred)
            val_loss = loss.item() * data.size(0)
            val_loss_hist.append(val_loss)

            loop2.set_description(f"Epoch [{epoch+1}/{num_epochs}] Validation")
            loop2.set_postfix(loss = loss.item())

        scheduler.step(val_loss)

        if min_val_loss > val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->\
                    {val_loss:6f}) \t Saving The Model')
            min_val_loss = val_loss
            torch.save(model.state_dict(), weights_name)

    print("Training Completed!")

    # create plots of loss over each iteration
    plt.figure()
    plt.plot(train_loss_hist, label="training loss")
    plt.plot(val_loss_hist, label="validation loss")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss function progress during training")
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, \
            help="path to generated training data m-file")
    parser.add_argument('--weights_name', type=str, \
            help="path to save model weights as hdf5-file")
    parser.add_argument('--meanstd_name', type=str, \
            help="path to save normalization factors as m-file")
    args = parser.parse_args()

    # train with given arguements
    train_model(abspath(args.filename), \
            abspath(args.weights_name), \
            abspath(args.meanstd_name))
