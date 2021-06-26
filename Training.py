import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import argparse
from os.path import abspath
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from CNN_Model import project_01, normalize_im, DeepSTORM, l1l2loss

device = ("cuda" if torch.cuda.is_available() else "cpu") 

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
    trainX, testX, trainY, testY = train_test_split(patches, heatmaps, \
            test_size=0.3,
            random_state=42)
    print(f'Number of Training Examples: {trainX.shape[0]}')
    print(f'Number of Validation Examples: {testX.shape[0]}')

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

    # label reshaping
    trainY = trainY.reshape(trainY.shape[0], 1, psize, psize)
    testY = testY.reshape(testY.shape[0], 1, psize, psize)

    # put data into PyTorch dataset
    train_dataset = SimpleSTORMDataset(trainX_norm, trainY)
    train_loader = DataLoader(train_dataset, \
            batch_size=16, shuffle=True)
    test_dataset = SimpleSTORMDataset(testX_norm, testY)
    test_loader = DataLoader(test_dataset, \
            batch_size=16, shuffle=True)

    # create Deep-STORM model
    model = DeepSTORM((psize, psize))
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
        loop1 = tqdm(train_loader, total=len(train_loader), leave=True)
        train_loss = 0
        model.train()
        for data, label in loop1:
            data.to(device)
            label.to(device)
            spikes_pred = model(data)
            loss = criterion(label, spikes_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item() * data.size(0)
            train_loss_hist.append(train_loss)

            loop1.set_description(f"Epoch [{epoch+1}/{num_epochs}] Training")
            loop1.set_postfix(loss = loss.item())

        loop2 = tqdm(test_loader, total=len(train_loader), leave=True)
        val_loss = 0
        model.eval()
        for data, label in loop2:
            data.to(device)
            label.to(device)
            spikes_pred = model(data)
            loss = criterion(label, spikes_pred)
            val_loss = loss.item() * data.size(0)
            val_loss_hist.append(val_loss)

            loop2.set_description(f"Epoch [{epoch+1}/{num_epochs}] Validation")
            loop2.set_postfix(loss = loss.item())

        scheduler.step(val_loss)

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f\
            }--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), weights_name)
    return


if __name__ == "__main__":

    # parse the required input arguements
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
