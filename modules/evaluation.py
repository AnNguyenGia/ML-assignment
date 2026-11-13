import torch
from sklearn.metrics import classification_report


def evaluate_torch_model(model, X, y, criterion, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features = torch.tensor(X, dtype=torch.float32, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        output = model(features)
        loss = criterion(output, labels)
        predictions = output.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

    report = classification_report(labels.cpu(), predictions.cpu())
    return loss.item(), accuracy, report