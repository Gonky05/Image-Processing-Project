import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Definição da Arquitetura da CNN 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
       
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return F.log_softmax(output, dim=1)


def treinar():
    
    batch_size = 64
    epochs = 5
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"A utilizar o dispositivo: {device}")

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    
    print("A carregar dataset MNIST...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

   # Inicializar modelo
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loop de Treino
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Época {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

    # Guardar o modelo treinado
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print(" Modelo guardado como 'mnist_cnn.pth'")

if __name__ == "__main__":
    treinar()