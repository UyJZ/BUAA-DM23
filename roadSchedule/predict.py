import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime

class Trajectory:
    def __init__(self, traj_id, data):
        self.traj_id = traj_id
        self.time_list = [datetime.strptime(item['time'], "%Y-%m-%dT%H:%M:%SZ") for item in data]
        self.entity_id_list = [item['entity_id'] for item in data]
        self.coordinates_list = [eval(item['coordinates']) for item in data]
        self.current_dis_list = [item['current_dis'] for item in data]
        self.speeds_list = [item['speeds'] for item in data]
        self.holidays_list = [item['holidays'] for item in data]

        # Calculate the time difference for each timestamp relative to the first timestamp
        self.duration_list = [(timestamp - self.time_list[0]).total_seconds() for timestamp in self.time_list]

    def __str__(self):
        return f"Trajectory {self.traj_id}: {len(self.time_list)} data points"

# Load JSON data and create Trajectory objects
with open('/mnt/f/codes/Python/BUAA-DM23/all_traj_data.json', 'r') as json_file:
    json_data = json.load(json_file)

# Convert Trajectory objects to PyTorch tensors
input_size = 7  # Replace with the actual input size
output_size = 7  # Replace with the actual output size

data_tensors = []

for traj in [Trajectory(item['traj_id'], item['data']) for item in json_data]:
    traj_data = []
    for i in range(len(traj.time_list) - 1):
        tensor_data = torch.tensor(
            [
                traj.duration_list[i],  # Use time difference instead of timestamp
                traj.entity_id_list[i],
                *traj.coordinates_list[i],  # Use * to unpack the coordinate list
                traj.current_dis_list[i],
                traj.speeds_list[i],
                traj.holidays_list[i]
            ],
            dtype=torch.float32
        )
        traj_data.append(tensor_data)

    # Pad the sequence to handle different lengths
    traj_padded = pad_sequence(traj_data, batch_first=True, padding_value=0)
    data_tensors.append(traj_padded)

# Pad the sequences along a new dimension for all trajectories
data_tensor = pad_sequence(data_tensors, batch_first=True, padding_value=0)

# Split the data into input (X) and target (y) sequences
X = data_tensor[:, :-1, :]  # Input sequence, excluding the last time step
y = data_tensor[:, 1:, :]   # Target sequence, excluding the first time step

# Move the data to GPU if available
if torch.cuda.is_available():
    X, y = X.cuda(), y.cuda()

# Create DataLoader for training data
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # Additional LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)  # Modify output_size to match input_size

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)  # Pass through the second LSTM layer
        out = self.fc(out)
        return out

# Instantiate the model and move it to GPU
hidden_size = 128  # Choose an appropriate hidden size
model = nn.DataParallel(LSTMModel(input_size, hidden_size, input_size)).cuda()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduce learning rate

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:, -1, :], targets[:, -1, :])  # Modify this line to only compare the last time step
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
