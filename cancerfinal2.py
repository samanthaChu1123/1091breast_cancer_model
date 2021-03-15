import torch
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

# seed(1)

# adjust hidden_node(H) = (5, 8, 11, 14) and epoch_num here

D_in, H, D_out = 9, 5, 1

epoch_num = 100
BATCH_SIZE = 10

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out, acti_func):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.act_func = acti_func
    
    def forward(self, x):
        h = self.linear1(x)
        h = self.act_func(h)
        y_pred = self.linear2(h)

        y_pred = torch.tanh(y_pred)
        return y_pred

# deal with file
print('reading file....')
train = pd.read_csv("breast_cancer_scal.txt", sep = " ", header=None)

X = train.values[0:, 2:11] # 683x9
y = train.values[0:, 0] # 683x1

'''
# 2多是良性 4惡性
plt.figure(figsize=(4,4))
train[0].value_counts().plot.bar(color=['black','green'])
plt.show()
'''

# str to float
row = range(683)
col = range(8)

for x in row:
    for k in col:
        X[x][k] = float(X[x][k][2:])
    X[x][8] = float(X[x][8][3:])

for n in row:
    if y[n] == 4:
        y[n] = -1
    else:
        y[n] = 1


X = X.astype('float64') 
y = y.astype('float64') 


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50)

print('Training data shape: ', X_train.shape, flush=True)
print('Training labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape, flush=True)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
print('Training data shape: ', X_train.shape, flush=True)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train).unsqueeze(1)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test).unsqueeze(1)
'''

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train).unsqueeze(1)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val).unsqueeze(1)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test).unsqueeze(1)


training_dataset = TensorDataset(X_train, y_train)
training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

testing_dataset = TensorDataset(X_test, y_test)
testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)

# adjust model here

model = TwoLayerNet(D_in, H, D_out, acti_func=torch.relu).double()

optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

loss_list = []
val_loss_list = []
val_acc_list = []

for batch_idx in range(epoch_num):

    total_loss = 0

    for x, y in tqdm(training_dataloader):

        y_pred = model(x)

        loss = torch.nn.functional.mse_loss(y_pred, y)

        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # start validation
    with torch.no_grad():#disable graident calculation during inferences

        val_loss = 0 # accumulate testing loss for each batch

        correct_cnt = 0.

        for x, y in val_dataloader:

            prediction = model(x) # call forward function

            # compute gradient of loss and update weights below
            loss = torch.nn.functional.mse_loss(prediction, y) # compute loss
            val_loss += loss

            prediction = (prediction >= 0)
            y = (y == 1.)

            # calculate accuracy (not sure if it will work, you can try)

            correct_cnt += ((prediction == y).sum())

        val_loss_list.append(val_loss)
        val_acc_list.append(correct_cnt)

    total_loss =  total_loss / len(training_dataloader)
    print('loss: ', total_loss, flush=True)
    loss_list.append(total_loss)


plt.plot(loss_list)
plt.plot(val_loss_list)
plt.title('loss',size = 20)
plt.show()

plt.plot(val_acc_list)
plt.title('validation set accuracy')
plt.show()


with torch.no_grad():#disable graident calculation during inferences

    test_loss = 0 # accumulate testing loss for each batch

    correct_cnt = 0.

    for x, y in testing_dataloader:

        prediction = model(x) # call forward function

        # compute gradient of loss and update weights below
        loss = torch.nn.functional.mse_loss(prediction, y) # compute loss
        test_loss += loss

        prediction = (prediction >= 0)
        y = (y == 1.)

        # calculate accuracy (not sure if it will work, you can try)

        correct_cnt += ((prediction == y).sum())
    
    # print("hidden layer = " + H + ", epoch_num = " + epoch_num)
    print("AVERAGE MSE LOSS W.R.T BATCH SIZE:{:.2f}, ACCURACY: {:.2f}".format(test_loss / len(testing_dataloader), 
                                                                            correct_cnt / len(testing_dataset)), flush=True)
        




