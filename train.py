import torchvision.models as models
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torchvision import transforms
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
#np.set_printoptions(threshold=sys.maxsize)

# load dataset and resize

dataset = np.load('dataset.npz')
GRID_SIZE = 8
GRIDS = 512 // GRID_SIZE
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.4)

train_dataset_maps = np.expand_dims(train_dataset['maps'], 1)
train_dataset_x = train_dataset['xpos'] // GRID_SIZE
train_dataset_y = train_dataset['ypos'] // GRID_SIZE
train_dataset_heading = train_dataset['heading']



trainingX = torch.tensor(train_dataset_maps)
trainingY = torch.tensor(np.zeros((train_dataset_maps.shape[0],)))#GRIDS * GRIDS)))
for i in range(train_dataset_maps.shape[0]):
  index = train_dataset_x[i] + 64 * train_dataset_y[i]
  trainingY[i] = index
 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, GRIDS * GRIDS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 16 * 125 * 125)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

# Load trained model weights
#model.load_state_dict(torch.load('/content/drive/My Drive/CSE571/weights.pth'))

criterion = nn.CrossEntropyLoss()  # Specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Specify which optimizer

epochs = 20
train_losses = []
indices = np.arange(600)
data = DataLoader(indices, batch_size=100,
                        shuffle=True, num_workers=4)
# Model Training
for j in range(epochs):
  for i, ind in enumerate(data):
      inputs, labels = trainingX[ind], trainingY[ind]
      y_pred = model(inputs)
      #print(labels[0:100].shape)
      loss = criterion(y_pred, labels.long())
      train_losses.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % 3 == 0:
          print('Epoch', i)
          print("Train loss", loss.item())
  print('Cycle', j)

#Save weights
#torch.save(model.state_dict(), '/content/drive/My Drive/CSE571/weights.pth')

# Visualize loss
x = [i for i in range(len(train_losses))]
plt.plot(x, train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# Plot prediction and original occupancy map for test dataset map 10
n = 10 # change this value to change the map and robot point you want the model to predict on
input = np.expand_dims(test_dataset['maps'][n], 0)
input = torch.tensor(np.expand_dims(input, 0))
output = model(input)
#output = np.argmax(output.data.numpy())

#x = output % 64
#y = output // 64
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
prob = output.data.numpy()
#prob = prob / np.linalg.norm(prob)
prob = (prob - np.min(prob))/np.ptp(prob)
for i in range(prob.shape[1]):
  if prob[0][i] < 0.9:
    prob[0][i] = 0
  #if prob[0][i] < -0.0001:
  #  prob[0][i] = 0
prob = prob.reshape((GRIDS, GRIDS))
prob = cv2.resize(prob, (512,512))
ax1.imshow(prob, cmap='hot')
x, y = dataset['xpos'][n], dataset['ypos'][n]
ax2.imshow(train_dataset_maps[n][0])
ax2.plot(x,y, 'ro')

plt.show()