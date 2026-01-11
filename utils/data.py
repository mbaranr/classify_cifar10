import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from torchvision.datasets import CIFAR10


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def display_image_grid(images, labels, classes, nrow=8):
    batch_size = images.size(0)
    ncol = (batch_size + nrow - 1) // nrow
    _, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    for i, img in enumerate(images):
        row = i // nrow
        col = i % nrow
        ax = axes[row, col] if ncol > 1 else axes[col]
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(classes[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_example_image(idx):
    ds = CIFAR10(
        root="../../assets/cifar10",
        train=True,
        download=True,
    )
    pil_img, label = ds[idx]
    pil_img = TF.resize(pil_img, (224, 224))
    tensor = TF.to_tensor(pil_img)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    img = TF.to_tensor(pil_img).permute(1, 2, 0)
    return tensor, img, label