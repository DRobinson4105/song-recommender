import torch
from torch import nn
from helper_functions import Autoencoder, processed_data
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = processed_data()
X_tensor = torch.FloatTensor(X).to(device)

model = Autoencoder(X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5000

for epoch in range(epochs):
    X_pred = model(X_tensor)

    loss = criterion(X_pred, X_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'{round(epoch / epochs * 100, 2)}%')

# Save model
torch.save(obj=model.state_dict(), f=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth'))