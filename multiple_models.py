import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FCNNModel(nn.Module):
    def __init__(self, num_in_features=31, num_out_features=11):
        super(FCNNModel, self).__init__()
        self.fc1 = nn.Linear(num_in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, 0:31].values
    y = data.iloc[:, 31:].values
    scaler_X = StandardScaler().fit(X)
    X_scaled = scaler_X.transform(X)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, epochs=100):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
        val_losses.append(total_val_loss / len(val_loader))
    return train_losses, val_losses


def plot_losses(train_losses_sets, val_losses_sets, title, filename, indices):
    plt.figure(figsize=(12, 8))
    for i in range(len(train_losses_sets)):
        plt.plot(train_losses_sets[i], label=f'Training Loss - Scenario {indices[0]}')
        plt.plot(val_losses_sets[i], '--', label=f'Validation Loss - Scenario {indices[1]}')

        # Save the loss values to a file
        train_loss_filename_1 = f'train_loss_scenario_{indices[0]}.txt'
        train_loss_filename_2 = f'train_loss_scenario_{indices[1]}.txt'
        val_loss_filename_1 = f'val_loss_scenario_{indices[0]}.txt'
        val_loss_filename_2 = f'val_loss_scenario_{indices[1]}.txt'

        with open(train_loss_filename_1, 'w') as f:
            for loss in train_losses_sets[0]:
                f.write(str(loss) + '\n')
        
        with open(train_loss_filename_2, 'w') as f:
            for loss in train_losses_sets[1]:
                f.write(str(loss) + '\n')

        with open(val_loss_filename_1, 'w') as f:
            for loss in val_losses_sets[0]:
                f.write(str(loss) + '\n')
        
        with open(val_loss_filename_2, 'w') as f:
            for loss in val_losses_sets[1]:
                f.write(str(loss) + '\n')

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()

def eval_model(model, test_loader):
    """
    Test the model and calculate MSE, MAE, and MAPE metrics.
    """
    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize criteria for MSE, MAE
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    # Initialize metric totals and counts
    total_loss_mse = 0.0
    total_loss_mae = 0.0
    total_loss_mape = 0.0
    total_count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate loss
            loss_mse = criterion_mse(outputs, targets)
            loss_mae = criterion_mae(outputs, targets)
            loss_mape = torch.mean(torch.abs((targets - outputs) / (targets + 1e-8))) * 100
            
            total_loss_mse += loss_mse.item()
            total_loss_mae += loss_mae.item()
            total_loss_mape += loss_mape.item()
            total_count += 1
    
    # Calculate average losses
    avg_loss_mse = total_loss_mse / total_count
    avg_loss_mae = total_loss_mae / total_count
    avg_loss_mape = total_loss_mape / total_count
    
    # Output test losses
    print(f'Test Loss (MSE): {avg_loss_mse}')
    print(f'Test Loss (MAE): {avg_loss_mae}')
    print(f'Test Loss (MAPE): {avg_loss_mape}')
    
    # Output the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')





if __name__ == "__main__":
    train_losses_all, val_losses_all = [], []
    for i in tqdm(range(1, 9)):
        data_path = f'/home/xinlin/raid/LLaMA-Factory/DISS/data/dataset{i}.csv'  # Update this path to your datasets' location
        train_loader, val_loader, test_loader = load_data(data_path)
        model = FCNNModel().to(device)
        save_path = f'model{i}.pth'
        print(f'Training on dataset {i}')
        train_losses, val_losses = train_model(model, train_loader, val_loader)
        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)
        # Save the trained model
        torch.save(model.state_dict(), save_path)

    # Create a directory to store the loss files
    # os.makedirs('loss_files', exist_ok=True)

    # plot_losses(train_losses_all[:2], val_losses_all[:2], 'Training and Validation Losses for Scenarios 1-2', 'figs/losses_1_to_2.pdf', [1, 2])
    # plot_losses(train_losses_all[2:4], val_losses_all[2:4], 'Training and Validation Losses for Scenarios 3-4', 'figs/losses_3_to_4.pdf', [3, 4])
    # plot_losses(train_losses_all[4:6], val_losses_all[4:6], 'Training and Validation Losses for Scenarios 5-6', 'figs/losses_5_to_6.pdf', [5, 6])
    # plot_losses(train_losses_all[6:], val_losses_all[6:], 'Training and Validation Losses for Scenarios 7-8', 'figs/losses_7_to_8.pdf', [7, 8])
