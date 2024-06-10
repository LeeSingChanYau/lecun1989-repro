"""

repro.py gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82

we can try to use our knowledge from 33 years later to improve on this,
but keeping the model size same.

Change 1: replace tanh on last layer with FC and use softmax. Had to
lower the learning rate to 0.01 as well. This improves the optimization
quite a lot, we now crush the training set:
23
eval: split train. loss 9.536698e-06. error 0.00%. misses: 0
eval: split test . loss 9.536698e-06. error 4.38%. misses: 87

Change 2: change from SGD to AdamW with LR 3e-4 because I find this
to be significantly more stable and requires little to no tuning. Also
double epochs to 46. I decay the LR to 1e-4 over course of training.
These changes make it so optimization is not culprit of bad performance
with high probability. We also seem to improve test set a bit:
46
eval: split train. loss 0.000000e+00. error 0.00%. misses: 0
eval: split test . loss 0.000000e+00. error 3.59%. misses: 72

Change 3: since we are overfitting we can introduce data augmentation,
e.g. let's intro a shift by at most 1 pixel in both x/y directions. Also
because we are augmenting we again want to bump up training time, e.g.
to 60 epochs:
60
eval: split train. loss 8.780676e-04. error 1.70%. misses: 123
eval: split test . loss 8.780676e-04. error 2.19%. misses: 43

Change 4: we want to add dropout at the layer with most parameters (H3),
but in addition we also have to shift the activation function to relu so
that dropout makes sense. We also bring up iterations to 80:
80
eval: split train. loss 2.601336e-03. error 1.47%. misses: 106
eval: split test . loss 2.601336e-03. error 1.59%. misses: 32

To be continued...
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter # pip install tensorboardX

# -----------------------------------------------------------------------------

class Net(nn.Module):
    """ 1989 LeCun ConvNet per description in the paper """

    def __init__(self):
        super().__init__()

        def kaiming_init(fan_in, *shape):
            std = (2.0 / fan_in)**0.5  
            return torch.randn(shape) * std

        # initialization as described in the paper to my best ability, but it doesn't look right...
        # self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
        # winit = lambda fan_in, *shape: (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in**0.5
        macs = 0 # keep track of MACs (multiply accumulates)
        acts = 0 # keep track of number of activations

        # H1 layer parameters and their initialization
        self.H1w = nn.Parameter(kaiming_init(5*5*1, 24, 1, 5, 5))
        self.H1b = nn.Parameter(torch.zeros(24, 8, 8)) # presumably init to zero for biases
        macs += (5*5*1) * (8*8) * 24
        acts += (8*8) * 24

        # H2 layer parameters and their initialization
        """
        H2 neurons all connect to only 8 of the 12 input planes, with an unspecified pattern
        I am going to assume the most sensible block pattern where 4 planes at a time connect
        to differently overlapping groups of 8/12 input planes. We will implement this with 3
        separate convolutions that we concatenate the results of.
        """
        self.H2w = nn.Parameter(kaiming_init(5*5*24, 24, 24, 5, 5))  # Assuming input and output channels are now 24
        self.H2b = nn.Parameter(torch.zeros(24, 4, 4))  # Adjust the size if the output dimension changes due to stride/padding
        macs += (5*5*24) * (4*4) * 24
        acts += (4*4) * 24

        # New H3w
        self.H3w = nn.Parameter(kaiming_init(5*5*24, 24, 24, 5, 5))  # Assuming input and output channels are now 24
        self.H3b = nn.Parameter(torch.zeros(24, 4, 4))  # Adjust the size if the output dimension changes due to stride/padding
        macs += (5*5*24) * (4*4) * 24
        acts += (4*4) * 24
        
        # H4 is a fully connected layer
        self.H4w = nn.Parameter(kaiming_init(4*4*24, 4*4*24, 30))  # Adjust based on actual output size from H3
        self.H4b = nn.Parameter(torch.zeros(30))
        macs += (4*4*24) * 30
        acts += 30

        # output layer is also fully connected layer
        self.outw = nn.Parameter(kaiming_init(30, 30, 62))
        self.outb = nn.Parameter(torch.zeros(62))
        macs += 30 * 62                 # (input_features * output_features)
        acts += 62

        self.macs = macs
        self.acts = acts

    def forward(self, x):

        # poor man's data augmentation by 1 pixel along x/y directions
        if self.training:
            shift_x, shift_y = np.random.randint(-1, 2, size=2)
            x = torch.roll(x, (shift_x, shift_y), (2, 3))

        # x has shape (1, 1, 16, 16)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        x = F.conv2d(x, self.H1w, stride=2) + self.H1b
        x = torch.relu(x)

        # x is now shape (1, 12, 8, 8)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0)
        x = F.conv2d(x, self.H2w, stride=2) + self.H2b
        x = torch.relu(x)

        # New H3 convolutional layer
        x = F.pad(x, (1, 1, 1, 1), 'constant', -1.0)  # Appropriate padding for the 3x3 kernel
        x = F.conv2d(x, self.H3w, padding=1, stride=1)  + self.H3b  # No stride specified, adjust if needed
        x = torch.relu(x)
        
        x = F.dropout(x, p=0.25, training=self.training)
    
        x = x.flatten(start_dim=1)  # Flattening the tensor for fully connected layer
        
        # First fully connected layer (previously H3, now H4)
        x = x @ self.H4w + self.H4b
        x = torch.relu(x)
    
        x = x @ self.outw + self.outb

        return x

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a 2022 but mini ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/modern', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))

    # init rng
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    # set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)

    # init a model
    model = Net()
    print("model stats:")
    print("# params:      ", sum(p.numel() for p in model.parameters())) # in paper total is 9,760
    print("# MACs:        ", model.macs)
    print("# activations: ", model.acts)

    # init data
    Xtr, Ytr = torch.load('emnist_train1989.pt')
    Xte, Yte = torch.load('emnist_test1989.pt')

    # init optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        model.eval()
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        Yhat = model(X)
        loss = F.cross_entropy(yhat, y.argmax(dim=1))
        err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
        print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}")
        writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
        writer.add_scalar(f'loss/{split}', loss.item(), pass_num)

    # train
    for pass_num in range(80):

        # learning rate decay
        alpha = pass_num / 79
        for g in optimizer.param_groups:
            g['lr'] = (1 - alpha) * args.learning_rate + alpha * (args.learning_rate / 3)

        # perform one epoch of training
        model.train()
        for step_num in range(Xtr.size(0)):

            # fetch a single example into a batch of 1
            x, y = Xtr[[step_num]], Ytr[[step_num]]

            # forward the model and the loss
            yhat = model(x)
            loss = F.cross_entropy(yhat, y.argmax(dim=1))

            # calculate the gradient and update the parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # after epoch epoch evaluate the train and test error / metrics
        print(pass_num + 1)
        eval_split('train')
        eval_split('test')

    # save final model to file
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model-alan.pt'))
