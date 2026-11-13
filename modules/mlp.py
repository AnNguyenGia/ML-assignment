import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        dims = [input_dim, 1024, 512, 256, 128, 64, 32]
        self.layers = nn.ModuleList([
            nn.Linear(dims[index], dims[index + 1])
            for index in range(len(dims) - 1)
        ])
        self.batchnorms = nn.ModuleList([
            nn.BatchNorm1d(dims[index + 1])
            for index in range(len(dims) - 1)
        ])
        self.output_layer = nn.Linear(dims[-1], output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.history_loss = []
        self.history_accuracy = []

    def forward(self, X):
        for linear, batchnorm in zip(self.layers, self.batchnorms):
            X = self.dropout(self.activation(linear(X)))
            X = batchnorm(X)
        return self.softmax(self.output_layer(X))

    def trainer(self, X, y, epochs=100, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        features = torch.tensor(X, dtype=torch.float32, device=device)
        labels = torch.tensor(y, dtype=torch.long, device=device)

        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            output = self(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            accuracy = (output.argmax(dim=1) == labels).float().mean().item()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")
            self.history_loss.append(loss.item())
            self.history_accuracy.append(accuracy)

        return self.history_loss, self.history_accuracy