import optuna
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from optuna.exceptions import TrialPruned
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
data_dir = Path("/kaggle/working/MO106/MO_106")
NUM_CLASSES = 106
NUM_WORKERS = 1

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    # Load model
    model = timm.create_model("mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True, num_classes=NUM_CLASSES)
    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model)
    model.to(device)

    # Data transforms
    data_config = timm.data.resolve_model_data_config(model)
    # transform = timm.data.create_transform(**data_config, is_training=True)

    train_transforms = timm.data.create_transform(**data_config, is_training=True, auto_augment="rand-m9-mstd0.5")
    test_transforms = timm.data.create_transform(**data_config, is_training=False)

    train_full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir, # target folder of images
        transform=train_transforms, # transforms to perform on data (images)
        target_transform=None)

    test_full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir, # target folder of images
        transform=test_transforms, # transforms to perform on data (images)
        target_transform=None)

    targets = [sample[1] for sample in train_full_dataset.samples]

    train_indices, test_indices = train_test_split(
        range(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    train_dataset = Subset(train_full_dataset, train_indices)
    test_dataset = Subset(test_full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5
    for epoch in range(EPOCHS):
        print("Epoch number: ", epoch)
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                correct += (pred.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_accuracy = correct / total

        # Report and check for pruning
        trial.report(val_accuracy, step=epoch)
        if trial.should_prune():
            raise TrialPruned()

    return val_accuracy

def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
