import matplotlib.pyplot as plt


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