import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LinearQNet(nn.Module):

    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, fileName='model.pth'):
        modelFolderPath = './model'

        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName)


class QTrainer:
    def __init(self, model, learningRate, gamma):
        self.model = model
        self.learningRate = learningRate
        self.gamma = gamma
        self.lossFunction = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), learningRate=self.learningRate)

    def trainStep(self, state, action, reward, nextState, gameOver):
        pass
