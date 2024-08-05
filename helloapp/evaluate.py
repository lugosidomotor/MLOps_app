import torch
from train import SimpleNN, CustomDataset
from torch.utils.data import DataLoader

def evaluate_model(test_file):
    dataset = CustomDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, prefetch_factor=2)

    model = SimpleNN(input_size=dataset.data.shape[1]-1)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    total_loss = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for features, label in dataloader:
            outputs = model(features)
            loss = criterion(outputs, label.view(-1, 1))
            total_loss += loss.item()

    print(f"Average Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    evaluate_model("test.csv")
