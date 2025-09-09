from model import PricePredictor
from custom_dataset import CustomDataset
import torch
from data_processing import X_test, X_train, y_train, y_test
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

print(X_train.iloc[0])
#print(y_train)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

joblib.dump(scaler_X, "scaler_X.pkl")

scaler_Y = StandardScaler()
y_train = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_Y.transform(y_test.values.reshape(-1, 1))

joblib.dump(scaler_Y, "scaler_Y.pkl")

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0)

model = PricePredictor(X_train.shape[1])
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    print(epoch)
    model.train()
    train_loss = 0.0
    for batch_features, batch_labes in train_loader:
        y_pred = model(batch_features)
        loss = loss_func(y_pred, batch_labes)
        train_loss += loss.item() * batch_features.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labes in test_loader:
            y_pred = model(batch_features)
            loss = loss_func(y_pred, batch_labes)
            val_loss += loss.item() * batch_features.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(test_loader.dataset)

    print(f"epoch {epoch}, traing_loss - {avg_train_loss}")
    print(f"epoch {epoch}, validation_loss - {avg_val_loss}")
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

print(train_losses[-1])
print(val_losses[-1])

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Per Epoch")
plt.legend()
plt.show()

torch.save(model.state_dict(), "model.pth")
