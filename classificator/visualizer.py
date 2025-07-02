import random
from PIL import Image
import matplotlib.pyplot as plt
import timm
from pathlib import Path
import torch
import torchvision
from typing import List


device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    
    target_image = torchvision.io.read_image(str(image_path), mode=torchvision.io.ImageReadMode.RGB).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    top5probs = torch.topk(target_image_pred_probs.flatten(), 5)
    for i in range(len(top5probs.values)):
        print(f"Pred: {class_names[top5probs.indices[i].cpu()]} | Prob: {top5probs.values[i].cpu():.3f}")

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

if __name__ == "__main__":
    model = timm.create_model(
        'mobilenetv4_conv_small.e2400_r224_in1k',
        pretrained=False,  # Start with random weights, since you'll load your own
        num_classes=100
    )
    checkpoint = torch.load("runs/512batch-10epoch-adam0.001/MobileNetV4-Mushroom.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    dataset_dir = Path("mushroom-dataset/")
    # image_path_list = list(dataset_dir.glob("*/*.jpg")) + list(dataset_dir.glob("*/*.jpeg")) + list(dataset_dir.glob("*/*.png"))

    # # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    dataset = torchvision.datasets.ImageFolder(
        root=dataset_dir, # target folder of images
        transform=transforms, # transforms to perform on data (images)
        target_transform=None)
    class_names = dataset.classes
    print(class_names)

    # plot_transformed_images(image_path_list, 
    #                     transform=transforms, 
    #                     n=3)
    pred_and_plot_image(model=model,
                    image_path="maslak1.jpg",
                    class_names=class_names,
                    transform=transforms,
                    device=device)