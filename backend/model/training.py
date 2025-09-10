from model import PricePredictor
from custom_dataset import CustomDataset
import torch
from data_processing import X_test, X_train, y_train, y_test
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

joblib.dump(scaler_X, "data/scaler_X.pkl")

scaler_Y = StandardScaler()
y_train = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_Y.transform(y_test.values.reshape(-1, 1))

joblib.dump(scaler_Y, "data/scaler_Y.pkl")

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=0)

model = PricePredictor(X_train.shape[1])
loss_func = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

epochs = 200
train_losses = []
val_losses = []


for epoch in range(epochs):
    all_preds = []
    all_prices = []
    real_error_all = []
    price_real = []
    price_pred = []

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
            all_preds.append(y_pred)
            all_prices.append(batch_labes)
            price_real.append(scaler_Y.inverse_transform(batch_labes))
            price_pred.append(scaler_Y.inverse_transform(y_pred))
            real_error_all.append((scaler_Y.inverse_transform(batch_labes)[0][0], abs(scaler_Y.inverse_transform(batch_labes)[0][0] - scaler_Y.inverse_transform(y_pred)[0][0])))


    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(test_loader.dataset)

    scheduler.step(avg_val_loss)

    print(f"epoch {epoch}, traing_loss - {avg_train_loss}")
    print(f"epoch {epoch}, validation_loss - {avg_val_loss}")

    all_preds = torch.cat(all_preds).numpy()
    all_reals = torch.cat(all_prices).numpy()

    real_preds = scaler_Y.inverse_transform(all_preds)
    real_reals = scaler_Y.inverse_transform(all_reals)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)


    #final_r2 = r2_score(real_reals, real_preds)
    #mse_error = mean_squared_error(real_reals, real_preds)
    #rmse_error = math.sqrt(mse_error)

    #for i in range(5):
     #   print(f"real-{price_real[i][0][0]}, pred-{price_pred[i][0][0]}")

    #print(f"r2_error-{final_r2}")
    #print(f"rmse_error-{rmse_error}")
#print(real_error_all)
#real_error_all.sort(key=lambda x:x[1])
#print(real_error_all[-5:])

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Per Epoch")
plt.legend()
plt.show()

torch.save(model.state_dict(), "data/model.pth")
