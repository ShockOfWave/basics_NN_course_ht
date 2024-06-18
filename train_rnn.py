import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


config = {
    'learning_rate': 1e-3,
    'epochs': 1000,
    'hidden_dim': 128,
    'output_dim': 1,
    'num_layers': 5,
}

wandb.init(project="steel_industry_rnn_ht", config=config, name="baseline_model")

# Load the dataset
file_path = 'Steel_industry_data.csv'
data = pd.read_csv(file_path)

# Preprocessing
data['date'] = pd.to_datetime(data['date'])
data['Hour'] = data['date'].dt.hour
data['Minute'] = data['date'].dt.minute
data['Day_of_week'] = LabelEncoder().fit_transform(data['Day_of_week'])
data['Load_Type'] = LabelEncoder().fit_transform(data['Load_Type'])
data['WeekStatus'] = LabelEncoder().fit_transform(data['WeekStatus'])

# Select features and target (excluding time-related features)
features = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
            'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM', 'WeekStatus',
            'Day_of_week', 'Load_Type', 'Hour', 'Minute']
target = 'Usage_kWh'

X = data[features].values
y = data[target].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define model class
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, model_type='RNN'):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, _ = self.rnn(x, (h0.detach(), c0.detach()))
        else:
            out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


# Training function
def train_model(model, X_train, y_train, X_test, y_test, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        outputs = outputs.detach().cpu().numpy()

        mae_train = mean_absolute_error(y_train, outputs)
        mse_train = mean_squared_error(y_train, outputs)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, outputs)

        wandb.log({
            'train_loss': loss.item(),
            'train_mae': mae_train,
            'train_mse': mse_train,
            'train_rmse': rmse_train,
            'train_r2': r2_train,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        with torch.inference_mode():
            model.eval()
            outputs = model(X_test)
            val_loss = criterion(outputs, y_test)

            outputs = outputs.detach().cpu().numpy()

            mae_val = mean_absolute_error(y_test, outputs)
            mse_val = mean_squared_error(y_test, outputs)
            rmse_val = np.sqrt(mse_val)
            r2_val = r2_score(y_test, outputs)

            wandb.log({
                'val_loss': val_loss.item(),
                'val_mae': mae_val,
                'val_mse': mse_val,
                'val_rmse': rmse_val,
                'val_r2': r2_val
            })

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}')

    model.eval()
    y_pred = model(X_test).detach().numpy()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2


# Define and train models
input_dim = X_train.shape[1]

# Reshape input tensors for RNN models
X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), 1, X_train_tensor.size(1))
X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), 1, X_test_tensor.size(1))

rnn_model = RNNModel(input_dim,
                     config['hidden_dim'],
                     config['output_dim'],
                     config['num_layers'],
                     model_type='RNN')
gru_model = RNNModel(input_dim,
                     config['hidden_dim'],
                     config['output_dim'],
                     config['num_layers'],
                     model_type='GRU')
lstm_model = RNNModel(input_dim,
                      config['hidden_dim'],
                      config['output_dim'],
                      config['num_layers'],
                      model_type='LSTM')

print("Training RNN...")
rnn_mse, rnn_rmse, rnn_r2 = train_model(rnn_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=config['epochs'])
print("Training GRU...")
gru_mse, gru_rmse, gru_r2 = train_model(gru_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=config['epochs'])
print("Training LSTM...")
lstm_mse, lstm_rmse, lstm_r2 = train_model(lstm_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=config['epochs'])

# Print the results
print(f"RNN - MSE: {rnn_mse:.4f}, RMSE: {rnn_rmse:.4f}, R^2: {rnn_r2:.4f}")
print(f"GRU - MSE: {gru_mse:.4f}, RMSE: {gru_rmse:.4f}, R^2: {gru_r2:.4f}")
print(f"LSTM - MSE: {lstm_mse:.4f}, RMSE: {lstm_rmse:.4f}, R^2: {lstm_r2:.4f}")
