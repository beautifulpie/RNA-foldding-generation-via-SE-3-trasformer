import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
from data_loader import ProteinTrajectoryDataset
from Model.model import DenoisingModel
import Model.loss_function as loss
# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize wandb
wandb.init(project="protein-md-trajectory", entity="your_wandb_username")
wandb.config.update(config)

# Load dataset
data_path = config["data_path"]
dataset = ProteinTrajectoryDataset(data_path)
data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingModel(dim=config["dim"], depth=config["depth"], num_tokens=config["num_tokens"]).to(device)
criterion = loss.total_loss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training loop
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0

    for sequences, structures, quaternions, translations in data_loader:
        sequences = sequences.to(device)
        structures = structures.to(device)
        quaternions = quaternions.to(device)
        translations = translations.to(device)
        
        pairwise_repr = None
        mask = None

        optimizer.zero_grad()
        outputs = model(sequences, structures, quaternions, translations, pairwise_repr, mask)
        loss = criterion(outputs, structures)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    average_loss = running_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {average_loss}')
    
    # Log the average loss to wandb
    wandb.log({"epoch": epoch+1, "loss": average_loss})

print('Finished Training')
