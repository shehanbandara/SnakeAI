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
        stateTensor = torch.tensor(state, dtype=torch.float)
        actionTensor = torch.tensor(action, dtype=torch.float)
        rewardTensor = torch.tensor(reward, dtype=torch.long)
        nextStateTensor = torch.tensor(nextState, dtype=torch.float)

        if len(stateTensor.shape) == 1:
            stateTensor = torch.unsqueeze(stateTensor, 0)
            actionTensor = torch.unsqueeze(actionTensor, 0)
            rewardTensor = torch.unsqueeze(rewardTensor, 0)
            nextStateTensor = torch.unsqueeze(nextStateTensor, 0)
            gameOver = (gameOver, )

        # Get predicted Q value with the current state
        prediction = self.model(stateTensor)

        # QNew = reward + gamma * max(next predicted Q value)
        predictionClone = prediction.clone()
        for i in range(len(gameOver)):
            QNew = rewardTensor[i]
            if not gameOver[i]:
                qNew = rewardTensor[i] + self.gamma * \
                    torch.max(self.model(nextStateTensor[i]))
            predictionClone[i][torch.argmax(actionTensor[i]).item()] = QNew

        self.optimizer.zero_grad()
        loss = self.criterion(predictionClone, prediction)
        loss.backward()

        self.optimizer.step()
