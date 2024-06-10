import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------

torch.manual_seed(1337)
np.random.seed(1337)

# Add a transform to convert images to tensors
transform = transforms.ToTensor()

for split in {'train', 'test'}:

    # Use the EMNIST dataset and include the transform
    data = datasets.EMNIST('./data', split='byclass', train=split=='train', download=True, transform=transform)

    # Assume n to be a smaller subset of the full dataset for quick experiments
    n = 7291 if split == 'train' else 2007
    rp = np.random.permutation(len(data))[:n]

    # Adjust the number of classes in the target tensor to 62
    X = torch.full((n, 1, 16, 16), 0.0, dtype=torch.float32)
    Y = torch.full((n, 62), -1.0, dtype=torch.float32)
    for i, ix in enumerate(rp):
        I, yint = data[int(ix)]  # Make sure to extract the image tensor from the dataset
        xi = I.unsqueeze(0)  # Ensure the image has a batch dimension if needed
        xi = F.interpolate(xi, size=(16, 16), mode='bilinear', align_corners=False)  # Correct interpolation
        X[i] = xi[0]  # Store the resized image
        Y[i, yint] = 1.0  # Set the correct class to have a target of +1.0

    torch.save((X, Y), f'emnist_{split}1989.pt')
