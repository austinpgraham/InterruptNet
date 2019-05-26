import torch.optim as optim
from torch import nn
from tensorly.decomposition import partial_tucker
import torchvision
import torch
import math
import tensorly
tensorly.set_backend('pytorch')


batch_size_train = 64
batch_size_test = 1000


def load_cifar_data():
    """
    Returns the generators for the training and testing data
    as given by PyTorch for the Cifar-10 data set
    """
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('cifarfiles/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])),
        batch_size=batch_size_train, shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('cifarfiles/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])),
        batch_size=batch_size_test, shuffle=True, pin_memory=True)
    return train_loader, test_loader


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv4_bn = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv7_bn = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv9 = nn.Conv2d(512, 512, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv9_bn = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3,
                                stride=1, padding=1, bias=True)
        self.conv10_bn = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3,
                                stride=1, padding=1, bias=True)
        self.conv11_bn = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*2*2, 512*2*2, bias=True)
        self.fc1_bn = nn.BatchNorm1d(512*2*2)
        self.relu1 = nn.ReLU(True)

        self.fc2 = nn.Linear(512*2*2, 512*2*2, bias=True)
        self.fc2_bn = nn.BatchNorm1d(512*2*2)
        self.relu2 = nn.ReLU(True)

        self.fc3 = nn.Linear(512*2*2, 1000, bias=True)
        self.fc3_bn = nn.BatchNorm1d(1000)
        self.relu3 = nn.ReLU(True)

        self.fc4 = nn.Linear(1000, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

        self.cuda()
        self._criterion = nn.CrossEntropyLoss().cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)
        self._log_interval = 10

    def plot_eigen_decomp(self, layer_str, epoch, plt):
        layer = getattr(self, layer_str)
        u, s, v = torch.svd(layer.weight)
        s = s.cpu().data.numpy()
        plt.plot(list(range(len(s))), s,
                 label="Weights at epoch {}".format(epoch))

    def current_spectral_norm(self, layer_str):
        layer = getattr(self, layer_str)
        u, s, v = torch.svd(layer.weight)
        s = s.cpu().data.numpy()
        return s[0]

    def compress_fc_layer(self, layer_int, rank_ratio, new_size):
        backward_layer = getattr(self, "fc{}".format(layer_int))
        forward_layer = getattr(self, "fc{}".format(layer_int + 1))
        batch_norm = getattr(self, "fc{}_bn".format(layer_int))

        def _rank_reduce(weights, rank_ratio):
            # Do not include the kernel dimensions
            eigen_length = min(weights.shape[:2])
            target_rank = [int(eigen_length * rank_ratio)]*2
            core, factors = partial_tucker(
                weights, modes=[0, 1], init="svd", svd="truncated_svd", rank=target_rank)
            return tensorly.tucker_to_tensor(core, factors)

        # Rank reduce both matrices
        backward = _rank_reduce(backward_layer.weight, rank_ratio)
        forward = _rank_reduce(forward_layer.weight, rank_ratio)

        # Minimum weight selction
        l2_norm = (torch.norm(forward, dim=0) ** 2) / forward.shape[0]
        _, idxs = l2_norm.topk(new_size)
        idxs, _ = idxs.sort()
        forward = forward[:, idxs]
        backward = backward[idxs]

        # Reset weight matrices
        new_backward_layer = nn.Linear(backward_layer.in_features, new_size)
        new_backward_layer.weight = nn.Parameter(backward)
        new_backward_layer.bias = nn.Parameter(backward_layer.bias.data[idxs])

        new_forward_layer = nn.Linear(new_size, forward_layer.out_features)
        new_forward_layer.weight = nn.Parameter(forward)
        new_forward_layer.bias = forward_layer.bias

        new_batch_norm = nn.BatchNorm1d(new_size)
        new_batch_norm.weight.data = batch_norm.weight.data[idxs]
        new_batch_norm.bias.data = batch_norm.bias.data[idxs]

        setattr(self, "fc{}".format(layer_int), new_backward_layer)
        setattr(self, "fc{}".format(layer_int + 1), new_forward_layer)
        setattr(self, "fc{}_bn".format(layer_int), new_batch_norm)

        self.cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)

    def compress_conv_layer(self, layer_int, rank_ratio, new_size):
        backward_layer = getattr(self, "conv{}".format(layer_int))
        forward_layer = getattr(self, "conv{}".format(layer_int + 1))
        batch_norm = getattr(self, "conv{}_bn".format(layer_int))

        def _rank_reduce(weights, rank_ratio):
            # Do not include the kernel dimensions
            eigen_length = min(weights.shape[:2])
            target_rank = [int(eigen_length * rank_ratio)]*2
            core, factors = partial_tucker(
                weights, modes=[0, 1], init="svd", svd="truncated_svd", rank=target_rank)
            return tensorly.tucker_to_tensor(core, factors)

        # Rank reduce both matrices
        backward = _rank_reduce(backward_layer.weight.data, rank_ratio)
        forward = _rank_reduce(forward_layer.weight.data, rank_ratio)

        # Calculate norm of each kernel
        def _reshape_norm(weights):
            # Sqaured norm of all kernels
            means = (torch.norm(weights, dim=(2, 3)) ** 2) / \
                (forward.shape[2] * forward.shape[3])
            # Squared norm of squared norm will be norm of vectorized kernels
            return (torch.norm(means, dim=0) ** 2) / forward.shape[0]
        forward_norm = _reshape_norm(forward)
        # Do selection

        def _select_idxs(weights, orig_mat, new_size, columns=False):
            _, idxs = weights.topk(new_size)
            idxs, _ = idxs.sort()
            return orig_mat[:, idxs, :, :] if columns else orig_mat[idxs, :, :, :], idxs
        forward, idxs = _select_idxs(
            forward_norm, forward, new_size, columns=True)
        backward = backward[idxs, :, :, :]

        # Reform layers
        new_backward_layer = nn.Conv2d(
            backward_layer.in_channels, new_size, kernel_size=3, stride=1, padding=1, bias=True)
        new_backward_layer.weight.data = backward
        new_backward_layer.bias.data = backward_layer.bias.data[idxs]

        new_forward_layer = nn.Conv2d(
            new_size, forward_layer.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        new_forward_layer.weight.data = forward
        new_forward_layer.bias.data = forward_layer.bias.data

        new_batch_norm = nn.BatchNorm2d(new_size)
        new_batch_norm.weight.data = batch_norm.weight.data[idxs]
        new_batch_norm.bias.data = batch_norm.bias.data[idxs]

        setattr(self, "conv{}".format(layer_int), new_backward_layer)
        setattr(self, "conv{}".format(layer_int + 1), new_forward_layer)
        setattr(self, "conv{}_bn".format(layer_int), new_batch_norm)

        self.cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)

    def expand_conv_layer(self, layer_int, new_size):
        backward_layer = getattr(self, "conv{}".format(layer_int))
        forward_layer = getattr(self, "conv{}".format(layer_int + 1))
        batch_norm = getattr(self, "conv{}_bn".format(layer_int))

        def _reshape_mean(weights):
            # Mean of all kernels
            means = torch.mean(weights, dim=(2, 3))
            # Mean of channel
            return torch.mean(means, dim=0)

        weight_avgs = _reshape_mean(forward_layer.weight.data)
        # Sort them for replication
        idxs = weight_avgs.argsort(descending=True)
        # Calculate multiplicative requirement
        extend_amount = (math.ceil(new_size / idxs.size()[0]))
        # Repeat the indices
        idxs = idxs.repeat(extend_amount)[:new_size]
        # Get divides
        _, inverse, ratios = idxs.unique(
            return_inverse=True, return_counts=True)
        ratios = ratios[inverse].float().repeat(extend_amount)[:new_size]
        ratios = ratios.unsqueeze(-1)
        # Chunk out to be sure we keep order correct
        SIZE = forward_layer.weight.shape[0]
        chunks = [idxs[SIZE*i:SIZE*i + SIZE].sort()[1] + (SIZE*i)
                  for i in range(extend_amount)]
        sorted_idxs = torch.cat(chunks)

        # Get and assign new weights
        new_l2_weights = forward_layer.weight[:, idxs, :, :]
        new_l2_weights = new_l2_weights / \
            ratios.t().unsqueeze(-1).unsqueeze(-1).expand_as(new_l2_weights)
        new_l1_weights = backward_layer.weight[idxs, :, :, :]

        # Reform layers
        new_backward_layer = nn.Conv2d(
            backward_layer.in_channels, new_size, kernel_size=3, stride=1, padding=1, bias=True)
        new_backward_layer.weight.data = new_l1_weights[sorted_idxs, :, :, :]
        new_backward_layer.bias.data = backward_layer.bias.data[idxs]

        new_forward_layer = nn.Conv2d(
            new_size, forward_layer.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        new_forward_layer.weight.data = new_l2_weights[:, sorted_idxs, :, :]
        new_forward_layer.bias.data = forward_layer.bias.data

        new_batch_norm = nn.BatchNorm2d(new_size)
        new_batch_norm.weight.data = batch_norm.weight.data[idxs]
        new_batch_norm.bias.data = batch_norm.bias.data[idxs]

        setattr(self, "conv{}".format(layer_int), new_backward_layer)
        setattr(self, "conv{}".format(layer_int + 1), new_forward_layer)
        setattr(self, "conv{}_bn".format(layer_int), new_batch_norm)

        self.cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)

    def expand_fc_layer(self, layer_int, new_size):
        """
        Expand a fully connected layer
        """
        backward_layer = getattr(self, "fc{}".format(layer_int))
        forward_layer = getattr(self, "fc{}".format(layer_int + 1))
        batch_norm = getattr(self, "fc{}_bn".format(layer_int))

        # Get averages, it should be transposed but we can go
        # along the other axis to make it easier
        weight_avgs = torch.mean(forward_layer.weight, dim=0)
        # Sort them for replication
        idxs = weight_avgs.argsort(descending=True)
        # Calculate multiplicative requirement
        extend_amount = (math.ceil(new_size / idxs.size()[0]))
        # Repeat the indices
        idxs = idxs.repeat(extend_amount)[:new_size]
        # Get divides
        _, inverse, ratios = idxs.unique(
            return_inverse=True, return_counts=True)
        ratios = ratios[inverse].float().repeat(extend_amount)[:new_size]
        ratios = ratios.unsqueeze(0)
        # Chunk out to be sure we keep order correct
        SIZE = forward_layer.weight.shape[1]
        chunks = [idxs[SIZE*i:SIZE*i + SIZE].sort()[1] + (SIZE*i)
                  for i in range(extend_amount)]
        sorted_idxs = torch.cat(chunks)
        # Get and assign new weights
        new_l2_weights = forward_layer.weight[:, idxs]
        new_l2_weights = new_l2_weights / ratios.expand_as(new_l2_weights)
        new_l1_weights = backward_layer.weight[idxs]

        # Reset weight matrices
        new_backward_layer = nn.Linear(backward_layer.in_features, new_size)
        new_backward_layer.weight = nn.Parameter(new_l1_weights[sorted_idxs])
        new_backward_layer.bias = nn.Parameter(backward_layer.bias.data[idxs])

        new_forward_layer = nn.Linear(new_size, forward_layer.out_features)
        new_forward_layer.weight = nn.Parameter(new_l2_weights[:, sorted_idxs])
        new_forward_layer.bias = forward_layer.bias

        new_batch_norm = nn.BatchNorm1d(new_size)
        new_batch_norm.weight.data = batch_norm.weight.data[idxs]
        new_batch_norm.bias.data = batch_norm.bias.data[idxs]

        setattr(self, "fc{}".format(layer_int), new_backward_layer)
        setattr(self, "fc{}".format(layer_int + 1), new_forward_layer)
        setattr(self, "fc{}_bn".format(layer_int), new_batch_norm)

        self.cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)

    def expand_conv_zero(self, layer_int, new_size):
        backward_layer = getattr(self, "conv{}".format(layer_int))
        forward_layer = getattr(self, "conv{}".format(layer_int + 1))
        batch_norm = getattr(self, "conv{}_bn".format(layer_int))

        # Get total number to generate
        fc2_shape = forward_layer.weight.shape
        expand_by = new_size - fc2_shape[1]
        zero_mat = torch.zeros([fc2_shape[0], expand_by, fc2_shape[2], fc2_shape[3]],
                               dtype=torch.float32).cuda()
        # Stack along columns. This avoid doing an unecessary transpose
        new_l2_weights = torch.cat(
            (forward_layer.weight.data, zero_mat), dim=1)

        # Do the same for layer one
        fc1_shape = backward_layer.weight.shape
        zero_mat = torch.zeros([expand_by, fc1_shape[1], fc1_shape[2], fc1_shape[3]],
                               dtype=torch.float32).cuda()
        # Stack backward by rows
        new_l1_weights = torch.cat(
            (backward_layer.weight.data, zero_mat), dim=0)

        new_bias = torch.cat([backward_layer.bias.data, torch.zeros(
            [expand_by], dtype=torch.float32).cuda()])
        new_backward_layer = nn.Conv2d(
            backward_layer.in_channels, new_size, kernel_size=3, stride=1, padding=1, bias=True)
        new_backward_layer.weight.data = new_l1_weights
        new_backward_layer.bias.data = new_bias

        new_forward_layer = nn.Conv2d(
            new_size, forward_layer.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        new_forward_layer.weight.data = new_l2_weights
        new_forward_layer.bias.data = forward_layer.bias.data

        new_weight = torch.cat([batch_norm.weight.data, torch.zeros(
            [expand_by], dtype=torch.float32).cuda()])
        new_bias = torch.cat([batch_norm.bias.data, torch.zeros(
            [expand_by], dtype=torch.float32).cuda()])

        new_batch_norm = nn.BatchNorm2d(new_size)
        new_batch_norm.weight.data = new_weight
        new_batch_norm.bias.data = new_bias

        setattr(self, "conv{}".format(layer_int), new_backward_layer)
        setattr(self, "conv{}".format(layer_int + 1), new_forward_layer)
        setattr(self, "conv{}_bn".format(layer_int), new_batch_norm)

        self.cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)

    def expand_fc_zero(self, layer_int, new_size):
        backward_layer = getattr(self, "fc{}".format(layer_int))
        forward_layer = getattr(self, "fc{}".format(layer_int + 1))
        batch_norm = getattr(self, "fc{}_bn".format(layer_int))

        # Get total number to generate
        fc2_shape = forward_layer.weight.shape
        expand_by = new_size - fc2_shape[1]
        zero_mat = torch.zeros([fc2_shape[0], expand_by],
                               dtype=torch.float32).cuda()
        # Stack along columns. This avoid doing an unecessary transpose
        new_l2_weights = torch.cat(
            (forward_layer.weight.data, zero_mat), dim=1)

        # Do the same for layer one
        fc1_shape = backward_layer.weight.shape
        zero_mat = torch.zeros([expand_by, fc1_shape[1]],
                               dtype=torch.float32).cuda()
        # Stack backward by rows
        new_l1_weights = torch.cat(
            (backward_layer.weight.data, zero_mat), dim=0)

        # Reset weight matrices
        new_backward_layer = nn.Linear(backward_layer.in_features, new_size)
        new_backward_layer.weight = nn.Parameter(new_l1_weights)
        new_bias = torch.cat([backward_layer.bias.data, torch.zeros(
            [expand_by], dtype=torch.float32).cuda()])
        new_backward_layer.bias = nn.Parameter(new_bias)

        new_forward_layer = nn.Linear(new_size, forward_layer.out_features)
        new_forward_layer.weight = nn.Parameter(new_l2_weights)
        new_forward_layer.bias = forward_layer.bias

        new_weight = torch.cat([batch_norm.weight.data, torch.zeros(
            [expand_by], dtype=torch.float32).cuda()])
        new_bias = torch.cat([batch_norm.bias.data, torch.zeros(
            [expand_by], dtype=torch.float32).cuda()])

        new_batch_norm = nn.BatchNorm1d(new_size)
        new_batch_norm.weight.data = new_weight
        new_batch_norm.bias.data = new_bias

        setattr(self, "fc{}".format(layer_int), new_backward_layer)
        setattr(self, "fc{}".format(layer_int + 1), new_forward_layer)
        setattr(self, "fc{}_bn".format(layer_int), new_batch_norm)

        self.cuda()
        self._optim = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, X):
        out = X
        for i in range(1, 12):
            conv = getattr(self, "conv{}".format(i))
            bn = getattr(self, "conv{}_bn".format(i))
            pool = getattr(self, "pool{}".format(i), None)
            out = nn.functional.relu(conv(out))
            if pool is not None:
                out = pool(out)
        out = out.view(-1, 512 * 2 * 2)
        for i in range(1, 4):
            layer = getattr(self, "fc{}".format(i))
            bn = getattr(self, "fc{}_bn".format(i))
            relu = getattr(self, "relu{}".format(i))
            out = bn(relu(layer(out)))
        return self.softmax(self.fc4(out))

    def train_net(self, epoch, train_loader, change=False):
        """
        Perform a training epoch
        """
        losses = []
        optimizer = self._optim
        cr = self._criterion
        first = None
        # For each batch, feed and calculate gradients
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            target = target.cuda()
            data = data.cuda()
            # Feed through network
            output = self(data)
            # Calculate loss
            loss = cr(output, target)
            # Perform Back prop
            loss.backward()
            optimizer.step()
            # Output loss
            if batch_idx % self._log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            losses.append(loss.item())
            if change:
                first = loss.item()
                change = False
        return losses, losses[0]

    def test_net(self, test_loader):
        """
        Test the network so far on the test set
        """
        test_loss = 0
        correct = 0
        cr = self._criterion
        # Make sure we don't modify the weights
        # while testing
        with torch.no_grad():
            for data, target in test_loader:
                data = data.cuda()
                target = target.cuda()
                # Feed the data
                output = self(data).cuda()
                # Calculate the loss
                test_loss += cr(output, target)
                # Get the predicted output and test whether or not
                # it aligns with the correct answer
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        # Output accuracy
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return 100. * correct / len(test_loader.dataset)
