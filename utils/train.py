import torch
from utils.plots import plot_losses
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    return avg_loss, accuracy

def train(model, num_epochs, train_loader, val_loader, optimizer, loss_fn, device):
    train_losses = []
    val_losses = []

    pbar = tqdm(range(num_epochs), desc="Training")

    for _ in pbar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc*100:.2f}%"
        })

    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    plot_losses([train_losses, val_losses], ["Train Loss", "Validation Loss"])