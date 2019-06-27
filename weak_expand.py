import torch
import tensorly

from network import SIZE
from network import TestNetwork

from matplotlib import pyplot as plt


# Generate the test sizes
TEST_SIZES = [100 * (x + 1) + SIZE for x in range(10)]
LABELS = ['Random Replacement', 'Zero Replacement', 'WeakExpand']


def plot_expand_results(random, ez, we, plt):
    diffs = [d - SIZE for d in TEST_SIZES[::-1]]
    plt.title.set_text('Mean Difference for WeakExpand')
    l1 = plt.plot(diffs, random[::-1], linestyle="dashed",
                  color="red", marker="o", label='Random Replacement')[0]
    l2 = plt.plot(diffs, ez[::-1], linestyle="dashed",
                  color="blue", marker="^", label='Zero Replacement')[0]
    l3 = plt.plot(diffs, we[::-1], linestyle="dashed",
                  color="green", marker="o", label='WeakExpand')[0]
    # plt.plot([d for d in diffs][::-1], rand_select, label="Minimum Weight")
    plt.set_xlabel('Neuron Delta')
    plt.set_ylabel('Norm of Difference in Activation')
    return [l1, l2, l3]


def plot_expand_results_var(random, ez, we, plt):
    diffs = [d - SIZE for d in TEST_SIZES[::-1]]
    plt.title.set_text('Variational Difference for WeakExpand')
    l1 = plt.plot(diffs, random[::-1], linestyle="dashed",
                  color="red", marker="o", label='Random Replacement')[0]
    l2 = plt.plot(diffs, ez[::-1], linestyle="dashed",
                  color="blue", marker="^", label='Zero Replacement')[0]
    l3 = plt.plot(diffs, we[::-1], linestyle="dashed",
                  color="green", marker="o", label='WeakExpand')[0]
    # plt.plot([d for d in diffs][::-1], rand_select, label="Minimum Weight")
    plt.set_xlabel('Neuron Delta')
    plt.set_ylabel('Norm of Difference in Activation')
    return [l1, l2, l3]


def select_gpu(idx):
    """
    Select an available GPU device.
    """
    print("Finding GPUs...")
    device_count = torch.cuda.device_count()
    if idx >= device_count:
        raise ValueError("GPU index above number of available GPUs")
    for i in range(device_count):
        _str = "Found {}: {}"
        name = torch.cuda.get_device_name(i)
        print(_str.format(i, name))
    print("Selected: {}".format(torch.cuda.get_device_name(idx)))
    return torch.cuda.device(idx)


def main():
    # TestNetwork.plot_eigen_decomp()
    with select_gpu(0) as _:
        print("Running simulations...")
        random = []
        random_var = []

        wexpand = []
        wexpand_var = []

        ezero = []
        ezero_var = []

        for size in TEST_SIZES:
            print("Running for size {}...".format(size))
            # Run random replacement iter
            print("Running random replacement...")
            rr_mean, rr_var = TestNetwork.run_random_replacement(size)
            random.append(rr_mean)
            random_var.append(rr_var)

            print("Running weak expand...")
            we_mean, we_var = TestNetwork.run_weak_expand(size)
            wexpand.append(we_mean)
            wexpand_var.append(we_var)

            print("Running expand by zeros...")
            ez, ez_var = TestNetwork.run_expand_zeros(size)
            ezero.append(ez)
            ezero_var.append(ez_var)

    print(TEST_SIZES)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    lines1 = plot_expand_results(random, ezero, wexpand, ax1)
    lines2 = plot_expand_results_var(random_var, ezero_var, wexpand_var, ax2)
    fig.legend(lines1 + lines2, labels=LABELS)
    plt.show()


if __name__ == '__main__':
    main()
