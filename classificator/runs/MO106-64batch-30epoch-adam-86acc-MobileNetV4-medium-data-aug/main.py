import torch
from torch import nn
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import torchvision

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# data_dir = Path("/kaggle/input/mushroom-species/dataset/")
data_dir = Path("/kaggle/working/MO106/MO_106")
# data_dir = Path("C:\praca-inzynierska\Mobile-Application-for-Mushroom-Species-Classification-Using-Neural-Networks\classificator\mushroom-dataset")

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        title (str): Title for the plot.
        cmap (matplotlib colormap): Colormap for the plot.
    """
    # Create a figure with a specified size
    plt.figure(figsize=figsize)
    
    # Display the confusion matrix as an image with a colormap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Define tick marks and labels for the classes on the axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # Label the axes
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Ensure the plot layout is tight
    plt.tight_layout()
    # Display the plot
    plt.show()

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module
              ):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context 
    y_test = []
    predictions = []
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            y_test.append(y)
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            predictions.append(test_pred_labels.cpu())
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, y_test, predictions

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          classes,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          ):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        test_loss, test_acc, y_test, predictions = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
        )
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
        save_model(model, "/kaggle/working/", f"Epoch-{epoch+1}-MobileNetV4-Mushroom.pth")

        # 6. Save train checkpoint
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"/kaggle/working/Epoch-{epoch+1}-checkpoint.pth.tar")

    y_test_flat = torch.cat(y_test).cpu().numpy()
    predictions_flat = torch.cat(predictions).cpu().numpy()
    
    cm = confusion_matrix(y_test_flat, predictions_flat)

    plot_confusion_matrix(cm, classes, figsize=(24, 22))
    print(classification_report(y_test_flat, predictions_flat, target_names=classes, digits=4))

    # 8. Return the filled results at the end of the epochs
    return results

def main():
    BATCH_SIZE = 64
    NUM_WORKERS = 1

    dataset_dir = data_dir

    model = timm.create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", 
        pretrained=True, 
        num_classes=106
    )

    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model)
    model.to(device)

    model.train()
 
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    train_transforms = timm.data.create_transform(**data_config, is_training=True, auto_augment="rand-m9-mstd0.5")
    test_transforms = timm.data.create_transform(**data_config, is_training=False)

    train_full_dataset = torchvision.datasets.ImageFolder(
        root=dataset_dir, # target folder of images
        transform=train_transforms, # transforms to perform on data (images)
        target_transform=None)

    test_full_dataset = torchvision.datasets.ImageFolder(
        root=dataset_dir, # target folder of images
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    
    # dataset = datasets.ImageFolder(
    #     root=dataset_dir, # target folder of images
    #     transform=transforms, # transforms to perform on data (images)
    #     target_transform=None)

    # targets = [sample[1] for sample in dataset.samples]

    # train_indices, test_indices = train_test_split(
    #     range(len(targets)),
    #     test_size=0.2,
    #     stratify=targets,
    #     random_state=42
    # )

    # train_dataset = Subset(dataset, train_indices)
    # test_dataset = Subset(dataset, test_indices)

    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     shuffle=True
    # )

    # test_dataloader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     shuffle=False
    # )

    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 10

    model = model.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=9.807292423382008e-05,
        weight_decay=0.0011281681948057559,
    )
    
    # Start the timer
    from timeit import default_timer as timer 
    print("Start training")
    start_time = timer()

    checkpoint = torch.load("/kaggle/working/Epoch-30-checkpoint.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Train model_0 
    results = train(model=model, 
                            train_dataloader=train_loader,
                            test_dataloader=test_loader,
                            optimizer=optimizer,
                            loss_fn=loss_fn, 
                            epochs=NUM_EPOCHS,
                            classes=train_full_dataset.classes)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    plt.plot(np.arange(len(results["train_loss"])), results["train_loss"], 'r', label = "Train")
    plt.plot(np.arange(len(results["test_loss"])), results["test_loss"], 'b', label = "Test")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(np.arange(len(results["train_acc"])), results["train_acc"], 'r', label = "Train")
    plt.plot(np.arange(len(results["test_acc"])), results["test_acc"], 'b', label = "Test")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
