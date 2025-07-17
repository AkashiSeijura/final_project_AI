import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import streamlit as st

# на уровне кода должна быть csv файл
try:
    data = pd.read_csv("vgsales.csv")
except Exception as e:
    print("Ошибка загрузки файла: vgsales.csv")
    print("Подробности:", e)
    exit()

# Предобработка данных
data = data.dropna(subset=['Genre', 'Platform', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])


features = data[['Genre', 'Platform', 'Year']]
targets = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]


label_encoder_genre = LabelEncoder()
label_encoder_platform = LabelEncoder()

features['Genre'] = label_encoder_genre.fit_transform(features['Genre'])
features['Platform'] = label_encoder_platform.fit_transform(features['Platform'])


scaler_year = StandardScaler()
scaler_sales = StandardScaler()

features['Year'] = scaler_year.fit_transform(features[['Year']])
targets = scaler_sales.fit_transform(targets.values)


X = torch.tensor(features.values, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class SalesPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(SalesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = SalesPredictor(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def compute_metrics(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.detach().numpy()
    mae = mean_absolute_error(y_true_np, y_pred_np, multioutput='raw_values')
    mse = mean_squared_error(y_true_np, y_pred_np, multioutput='raw_values')
    r2 = r2_score(y_true_np, y_pred_np, multioutput='raw_values')
    return mae, mse, r2

# Обучение модели
num_epochs = 100
train_mae, train_mse, train_r2 = [], [], []
test_mae, test_mse, test_r2 = [], [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()


    mae, mse, r2 = compute_metrics(y_train, outputs)
    train_mae.append(mae.mean())
    train_mse.append(mse.mean())
    train_r2.append(r2.mean())


    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        mae, mse, r2 = compute_metrics(y_test, test_outputs)
        test_mae.append(mae.mean())
        test_mse.append(mse.mean())
        test_r2.append(r2.mean())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'sales_predictor.pth')

# Загрузка модели
model.load_state_dict(torch.load('sales_predictor.pth'))
model.eval()

# Визуализация метрик
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot(train_mae, label='Train MAE')
ax1.plot(test_mae, label='Test MAE')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MAE')
ax1.set_title('Mean Absolute Error')
ax1.legend()

ax2.plot(train_mse, label='Train MSE')
ax2.plot(test_mse, label='Test MSE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.set_title('Mean Squared Error')
ax2.legend()

ax3.plot(train_r2, label='Train R²')
ax3.plot(test_r2, label='Test R²')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('R²')
ax3.set_title('R² Score')
ax3.legend()

plt.tight_layout()
plt.show()

# Вычисление важности признаков через Permutation Importance
def permutation_importance_pytorch(model, X, y, metric, n_repeats=10):
    model.eval()
    baseline_score = metric(y.numpy(), model(X).detach().numpy(), multioutput='raw_values').mean()
    importances = np.zeros(X.shape[1])

    for col in range(X.shape[1]):
        scores = []
        X_temp = X.clone()
        for _ in range(n_repeats):
            np.random.shuffle(X_temp[:, col].numpy())
            scores.append(metric(y.numpy(), model(X_temp).detach().numpy(), multioutput='raw_values').mean())
        importances[col] = baseline_score - np.mean(scores)

    return importances

# Вычисляем важность признаков
importances = permutation_importance_pytorch(model, X_test, y_test, mean_squared_error)
feature_names = ['Genre', 'Platform', 'Year']
for name, imp in zip(feature_names, importances):
    print(f'Feature: {name}, Importance: {imp:.4f}')

# Streamlit-интерфейс
st.title('Video Game Sales Prediction')

genres = label_encoder_genre.classes_
platforms = label_encoder_platform.classes_
genre = st.selectbox('Select Genre', genres)
platform = st.selectbox('Select Platform', platforms)
year = st.slider('Select Year', min_value=1980, max_value=2020, value=2017)

example = pd.DataFrame({
    'Genre': [genre],
    'Platform': [platform],
    'Year': [year]
})

example['Genre'] = label_encoder_genre.transform(example['Genre'])
example['Platform'] = label_encoder_platform.transform(example['Platform'])
example['Year'] = scaler_year.transform(example[['Year']])
example_tensor = torch.tensor(example.values, dtype=torch.float32)

with torch.no_grad():
    pred = model(example_tensor)
    pred_sales = scaler_sales.inverse_transform(pred.numpy())
    st.write(f'Predicted Sales (in millions):')
    st.write(f'NA Sales: {pred_sales[0][0]:.2f}')
    st.write(f'EU Sales: {pred_sales[0][1]:.2f}')
    st.write(f'JP Sales: {pred_sales[0][2]:.2f}')
    st.write(f'Other Sales: {pred_sales[0][3]:.2f}')
