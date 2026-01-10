import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms

torch.set_num_threads(1)

if __name__ == "__main__":
    ds = CIFAR10(
        root="../../assets/cifar10", 
        train=True, 
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    times = {}
    num_epochs = 6  
    batch_size = 64

    for num_workers in [0, 2, 4, 8]:
        
        print(f"Running with {num_workers} workers...")

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        epoch_times = []

        for epoch in range(num_epochs):
            start_time = time.perf_counter()

            for _, _ in loader:
                pass

            end_time = time.perf_counter()
            epoch_time = (end_time - start_time) * 1000

            # skip warm-up epoch
            if epoch > 0:
                epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        times[num_workers] = avg_time
        print(f"Num Workers: {num_workers}, Avg Epoch Time: {avg_time:.2f} milliseconds")

    workers = sorted(times.keys())
    epoch_times = [times[w] for w in workers]

    plt.figure(figsize=(6, 4))
    plt.plot(workers, epoch_times, marker="o")
    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Average Time per Epoch (milliseconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("num_workers_vs_epoch_time.png")
    plt.close()