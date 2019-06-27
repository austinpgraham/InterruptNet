import matplotlib.pyplot as plt
from tensorly.decomposition import partial_tucker
import math
import torch
import tensorly
tensorly.set_backend('pytorch')


ITERATIONS = 10
SIZE = 2048
TEST_INPUT_SIZE = 1000


class TestNetwork(torch.nn.Module):
    """
    This is a small three layer network used for isolation testing.
    It simply implements a FC neural network to test transformations.
    """

    def __init__(self):
        super(TestNetwork, self).__init__()
        # The sizes for each of these layers were chosen to reflect sizes
        # in the object detection network VGG16.
        # No bias is included for simplicity.
        self.fc1 = torch.nn.Linear(SIZE, SIZE, bias=False)
        # Using the tanH function because it is more popular.
        self.tanh1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(SIZE, SIZE, bias=False)
        self.tanh2 = torch.nn.Tanh()
        # Run on GPU.
        self.cuda()

    def forward(self, X):
        """
        Feed an input forward
        """
        out = X
        out = self.tanh1(self.fc1(out))
        return self.tanh2(self.fc2(out))

    def compress(self, new_size, rank_ratio):
        """
        Perform EigenRemove
        """

        def _rank_reduce(weights, rank_ratio):
            # Do not include the kernel dimensions
            eigen_length = min(weights.shape[:2])
            target_rank = [int(eigen_length * rank_ratio)]*2
            core, factors = partial_tucker(
                weights, modes=[0, 1], init="svd", svd="truncated_svd", rank=target_rank)
            return tensorly.tucker_to_tensor(core, factors)

        reduced = _rank_reduce(self.fc2.weight, rank_ratio)

        # Minimum weight selection
        l2_norm = (torch.norm(reduced, dim=0)**2) / reduced.shape[0]
        _, idxs = l2_norm.topk(new_size)
        idxs, _ = idxs.sort()
        selected_l2 = reduced[:, idxs]

        reduced = _rank_reduce(self.fc1.weight, rank_ratio)

        # Minimum weight seelection
        selected_l1 = reduced[idxs]

        # Reset weight matrices
        self.fc2 = torch.nn.Linear(new_size, SIZE)
        self.fc2.weight = torch.nn.Parameter(selected_l2)

        self.fc1 = torch.nn.Linear(SIZE, new_size)
        self.fc1.weight = torch.nn.Parameter(selected_l1)
        self.cuda()

    def expand(self, new_size, *args):
        """
        Perform WeakExpand
        """
        # Get averages, it should be transposed but we can go
        # along the other axis to make it easier
        weight_avgs = torch.mean(self.fc2.weight, dim=0)
        # Sort them for replication
        idxs = weight_avgs.argsort(descending=True)
        # Calculate multiplicative requirement
        extend_amount = (math.ceil(new_size / idxs.size()[0]))
        short_amount = idxs.size()[0] * (extend_amount - 1)
        # Repeat the indices
        idxs = idxs.repeat(extend_amount)[:new_size]
        # Get divides
        _, inverse, ratios = idxs.unique(
            return_inverse=True, return_counts=True)
        ratios = ratios[inverse].float().repeat(extend_amount)[:new_size]
        ratios = ratios.unsqueeze(0)
        # Chunk out to be sure we keep order correct
        chunks = [idxs[SIZE*i:SIZE*i + SIZE].sort()[1] + (SIZE*i)
                  for i in range(extend_amount)]
        sorted_idxs = torch.cat(chunks)
        # Get and assign new weights
        new_l2_weights = self.fc2.weight[:, idxs]
        new_l2_weights = new_l2_weights / ratios.expand_as(new_l2_weights)
        new_l1_weights = self.fc1.weight[idxs]
        # Reset weight matrices
        self.fc2 = torch.nn.Linear(new_size, SIZE)
        self.fc2.weight = torch.nn.Parameter(new_l2_weights[:, sorted_idxs])

        self.fc1 = torch.nn.Linear(SIZE, new_size)
        self.fc1.weight = torch.nn.Parameter(new_l1_weights[sorted_idxs])
        self.cuda()

    def expand_zeros(self, new_size, *args):
        """
        Expand the layer weights by adding almost zeros
        """
        # Get total number to generate
        fc2_shape = self.fc2.weight.shape
        expand_by = new_size - fc2_shape[1]
        zero_mat = torch.zeros([fc2_shape[0], expand_by],
                               dtype=torch.float32).cuda()
        # Stack along columns. This avoid doing an unecessary transpose
        new_l2_weights = torch.cat((self.fc2.weight.data, zero_mat), dim=1)

        # Do the same for layer one
        fc1_shape = self.fc1.weight.shape
        zero_mat = torch.zeros([expand_by, fc1_shape[1]],
                               dtype=torch.float32).cuda()
        # Stack backward by rows
        new_l1_weights = torch.cat((self.fc1.weight.data, zero_mat), dim=0)
        # Reset weight matrices
        self.fc2 = torch.nn.Linear(new_size, SIZE)
        self.fc2.weight = torch.nn.Parameter(new_l2_weights)

        self.fc1 = torch.nn.Linear(SIZE, new_size)
        self.fc1.weight = torch.nn.Parameter(new_l1_weights)
        self.cuda()

    def random_replace(self, new_size, *args):
        """
        Randomly replace weights
        """
        self.fc1 = torch.nn.Linear(SIZE, new_size)
        self.fc2 = torch.nn.Linear(new_size, SIZE)
        self.cuda()

    def min_weight_select(self, new_size, *args):
        """
        Perform minimum weight selection. This is done by measuring
        the L2 norm of the weight set.

        The only weights considered here are those being output
        from the neuron. This is generalized and discussed further in
        Molchanov et. al 2017.
        """
        l2_norm = torch.norm(self.fc2.weight, dim=0)
        _, idxs = l2_norm.topk(new_size)
        idxs, _ = idxs.sort()
        selected_l2 = self.fc2.weight[:, idxs]
        selected_l1 = self.fc1.weight[idxs]

        self.fc2 = torch.nn.Linear(new_size, SIZE)
        self.fc2.weight = torch.nn.Parameter(selected_l2)

        self.fc1 = torch.nn.Linear(SIZE, new_size)
        self.fc1.weight = torch.nn.Parameter(selected_l1)
        self.cuda()

    @classmethod
    def _run_exp(cls, new_size, _func, *args):
        results = []
        for i in range(ITERATIONS):
            print("Running iteration {}...".format(i + 1))
            # ITERATION times, run the test input vectors,
            # collect the output, do compression with new_size,
            # run again, measure results.
            test_nn = TestNetwork()
            test_inputs = torch.FloatTensor(
                TEST_INPUT_SIZE, SIZE).uniform_(0, 1).cuda()
            orig_output = test_nn(test_inputs)
            # Do the targeted operation
            _func_exec = getattr(test_nn, _func)
            _func_exec(new_size, *args)
            new_output = test_nn(test_inputs)
            # Calculate norm and add to results
            results.append(torch.norm(orig_output - new_output))
        results = torch.stack(results)
        return torch.mean(results).cpu().data.numpy(), torch.var(results).cpu().data.numpy()

    @classmethod
    def run_compress_iter(cls, new_size, rank_ratio):
        return cls._run_exp(new_size, 'compress', rank_ratio)

    @classmethod
    def run_random_replacement(cls, new_size):
        return cls._run_exp(new_size, 'random_replace')

    @classmethod
    def run_min_weight(cls, new_size):
        return cls._run_exp(new_size, 'min_weight_select')

    @classmethod
    def run_weak_expand(cls, new_size):
        return cls._run_exp(new_size, 'expand')

    @classmethod
    def run_expand_zeros(cls, new_size):
        return cls._run_exp(new_size, 'expand_zeros')

    @classmethod
    def plot_eigen_decomp(cls):
        test_nn = TestNetwork()
        u, s, v = torch.svd(test_nn.fc1.weight)
        s = s.cpu().data.numpy()
        plt.plot(list(range(len(s))), s, label="Backward Layer")
        u, s, v = torch.svd(test_nn.fc2.weight)
        s = s.cpu().data.numpy()
        plt.plot(list(range(len(s))), s, label="Forward Layer")
        plt.title('Singular Values of Layers')
        plt.legend()
        plt.ylabel('Value')
        plt.xlabel('Order')
        plt.show()
