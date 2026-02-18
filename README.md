import numpy as np
import sklearn 
from sklearn.datasets import make_circles
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# CREATING A SAMPLE TO STUDY
n_samples= 1000

# create circles 
X, Y = make_circles(n_samples,
                    noise= 0.03,
                    random_state=42 )

# visualization of data
plt.scatter(X[:, 0],
            X[:, 1],
            c=Y,
            cmap=plt.cm.RdYlBu)

plt.show()

# The data sets are in form of NumPy format, so we have to convert the-
# -data set in terms of tensors as we are dealing with Pytorch

X_pytorch= torch.from_numpy(X).type(torch.float)
y_pytorch= torch.from_numpy(Y).type(torch.float)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,
                                                   test_size= 0.2,
                                                   random_state= 42)
# Model building
class circlemodel(nn.Module):
    def __init__(self):
        super().__init__()

        # Using Sequential API for layers

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layers(x)

# Checking for device agnostic code

device = "xpu" if torch.xpu.is_available() else "cpu"

model_0 = circlemodel().to(device)

X_train_pytorch= torch.from_numpy(X_train).type(torch.float).to(device)
Y_train_pytorch= torch.from_numpy(Y_train).type(torch.float).to(device)

X_test_pytorch= torch.from_numpy(X_test).type(torch.float).to(device)
y_test_pytorch= torch.from_numpy(Y_test).type(torch.float).to(device)

# predictions 

with torch.inference_mode():    
    y_prde= model_0(X_test_pytorch)

# Set up loss function 
loss_function= nn.BCEWithLogitsLoss()

# Set up optimizer
optimizer= torch.optim.Adam(params= model_0.parameters(), lr= 0.01)

# Accuracy function 

def accuracy_fn(y_test_pytorch, y_prde):
    correct= torch.eq(y_test_pytorch, y_prde).sum().item()
    acc= (correct/len(y_prde))*100
    return acc

# Using a sigmoid activation function to turn them into prediction probabilities 
y_pred_prob= torch.sigmoid(y_prde)

# Converting prediction probabilities to labels
y_pred_round= torch.round(y_pred_prob)

# Training loop 

import tqdm as tqdm 
epochs = 50000
for epoch in tqdm(range(epochs)):

    model_0.train()

    # Forward pass on TRAIN DATA
    y_prd = model_0(X_train_pytorch)

    # Compute loss
    loss = loss_function(y_prd.squeeze(), Y_train_pytorch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- Evaluation ---
    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(X_test_pytorch)
        test_pred_prob = torch.sigmoid(test_pred)
        test_pred_round = torch.round(test_pred_prob)

    acc = accuracy_fn(y_test_pytorch, test_pred_round.squeeze())

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test Accuracy: {acc:.2f}%")


def plot_decision_boundary(model, X, y):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Visualize decision boundary after training
model_0.to("cpu")
plot_decision_boundary(model_0, X_pytorch, y_pytorch)
plt.show()
