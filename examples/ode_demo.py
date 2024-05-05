# Import necessary libraries
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set up command-line argument parsing
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

# Choose the integration method based on command line argument
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# Configure the device to use for computation (GPU or CPU)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Initialize the true initial condition and system parameters
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

# Define a simple neural network module to represent an ODE system
class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)

# Solve the ODE using the specified method without training gradients
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')

# Function to get a random batch of data
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # initial conditions for the batch
    batch_t = t[:args.batch_time]  # time steps for the batch
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # solutions for the batch
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

# Ensure the directory for saving visualizations exists
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs('png')

# Function to visualize the trajectories and phase portraits
def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = plt.subplot(131, frameon=False)
        ax_phase = plt.subplot(132, frameon=False)
        ax_vecfield = plt.subplot(133, frameon=False)
        
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], label='True Trajectory y1', color='green')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], label='True Trajectory y2', color='green', linestyle='dotted')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], label='Predicted Trajectory y1', linestyle='--', color='blue')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], label='Predicted Trajectory y2', linestyle='--', color='blue')
        ax_traj.legend()

        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label='True Phase')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--', label='Predicted Phase')
        ax_phase.legend()

        plt.tight_layout()
        plt.savefig(f'png/{itr:03d}.png')  # Save the image
        plt.close()  # Close the figure to free up memory

# Main model definition
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

    def forward(self, t, y):
        return self.net(y**3)

# Class for tracking the running average of loss or time
class RunningAverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

# Main execution block
if __name__ == '__main__':
    ii = 0
    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print(f'Iter {itr:04d} | Total Loss {loss:.6f}')
                visualize(true_y, pred_y, func, itr)

        end = time.time()
