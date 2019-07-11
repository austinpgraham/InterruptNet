import math
import torch
import tensorly

from scipy import stats

from network import SIZE
from network import TestNetwork

from matplotlib import pyplot as plt


# Generate the test sizes
TEST_SIZES = [100 * (x + 1) for x in range(10)]
LABELS = ['Random Replacement', 'EigenRemove, Rank=Full', 'EigenRemove, Rank=3/4',
          'EigenRemove, Rank=1/2', 'EigenRemove, Rank=1/4', 'EigenRemove, Rank=1/8',
          'Minimum Weight Selection']


def plot_compression_results(random, run_se, run_tf, run_half, run_fourth, run_eight, min_weight, plt):
    diffs = [SIZE - d for d in TEST_SIZES[::-1]]
    plt.title.set_text('Mean of Norm Difference for EigenRemove')
    l1 = plt.errorbar(diffs, random[0][::-1], yerr=[math.sqrt(x)
                                                    for x in random[1][::-1]], label='Random Replacement')[0]
    l2 = plt.errorbar(diffs, run_se[0][::-1], yerr=[math.sqrt(x) for x in run_se[1][::-1]], marker='o',
                      label='EigenRemove, Rank=Full')[0]
    l3 = plt.errorbar(diffs, run_tf[0][::-1], yerr=[math.sqrt(x) for x in run_tf[1][::-1]], color="darkgoldenrod",
                      linestyle='dashed', marker='o', label='EigenRemove, Rank=3/4')[0]
    l4 = plt.errorbar(diffs, run_half[0][::-1], yerr=[math.sqrt(x) for x in run_half[1][::-1]], color='hotpink',
                      linestyle='dashed', marker='^', label='EigenRemove, Rank=1/2')[0]
    l5 = plt.errorbar(diffs, run_fourth[0][::-1], yerr=[math.sqrt(x) for x in run_fourth[1][::-1]],
                      color='cadetblue', marker='o', label='EigenRemove, Rank=1/4')[0]
    l6 = plt.errorbar(diffs, run_eight[0][::-1], yerr=[math.sqrt(x) for x in run_eight[1][::-1]],
                      color='green', marker='^', label='EigenRemove, Rank=1/8')[0]
    l7 = plt.errorbar(diffs, min_weight[0][::-1], linestyle='dashed', yerr=[math.sqrt(x) for x in min_weight[1][::-1]],
                      color='red', marker='o', label='Minimum Weight Selection')[0]
    # plt.plot([d for d in diffs][::-1], rand_select, label="Minimum Weight")
    plt.set_xlabel('Neuron Delta')
    plt.set_ylabel('Norm of Difference in Activation')
    return [l1, l2, l3, l4, l5, l6, l7]


def plot_compression_results_var(random, run_se, run_tf, run_half, run_fourth, run_eight, min_weight, plt):
    diffs = [SIZE - d for d in TEST_SIZES[::-1]]
    plt.title.set_text('Variance of Norm Difference for EigenRemove')
    l1 = plt.plot(diffs, random[::-1], label='Random Replacement')[0]
    l2 = plt.plot(diffs, run_se[::-1], 'C2-o',
                  label='EigenRemove, Rank=Full')[0]
    l3 = plt.plot(diffs, run_tf[::-1], color="darkgoldenrod",
                  linestyle='dashed', marker='o', label='EigenRemove, Rank=3/4')[0]
    l4 = plt.plot(diffs, run_half[::-1], color='hotpink',
                  linestyle='dashed', marker='^', label='EigenRemove, Rank=1/2')[0]
    l5 = plt.plot(diffs, run_fourth[::-1],
                  color='cadetblue', marker='o', label='EigenRemove, Rank=1/4')[0]
    l6 = plt.plot(diffs, run_eight[::-1],
                  color='green', marker='^', label='EigenRemove, Rank=1/8')[0]
    l7 = plt.plot(diffs, min_weight[::-1], linestyle='dashed',
                  color='red', marker='o', label='Minimum Weight Selection')[0]
    # plt.plot([d for d in diffs][::-1], rand_select, label="Minimum Weight")
    plt.set_xlabel('Neuron Delta')
    plt.set_ylabel('Norm of Difference in Activation')
    return [l1, l2, l3, l4, l5, l6, l7]


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

        run_seven_eights = []
        run_full_var = []

        run_three_fourths = []
        run_tf_var = []

        run_half = []
        run_half_var = []

        run_fourth = []
        run_fourth_var = []

        run_eigth = []
        run_eigth_var = []

        min_weight = []
        min_weight_var = []
        for size in TEST_SIZES:
            if size == 0:
                size = 1
            print("Running for size {}...".format(size))
            # Run random replacement iter
            print("Running random replacement...")
            rr_mean, rr_var = TestNetwork.run_random_replacement(size)
            random.append(rr_mean)
            random_var.append(rr_var)

            print("Running compression with full rank...")
            full_mean, full_var = TestNetwork.run_compress_iter(size, 1)
            run_seven_eights.append(full_mean)
            run_full_var.append(full_var)

            print("Running compression 3/4 rank...")
            tf_mean, tf_var = TestNetwork.run_compress_iter(size, 3/4)
            run_three_fourths.append(tf_mean)
            run_tf_var.append(tf_var)

            print("Running compression 1/2 rank...")
            half_mean, half_var = TestNetwork.run_compress_iter(size, 0.5)
            run_half.append(half_mean)
            run_half_var.append(half_var)

            print("Running compression 1/4 rank...")
            fourth_mean, fourth_var = TestNetwork.run_compress_iter(size, 0.25)
            run_fourth.append(fourth_mean)
            run_fourth_var.append(fourth_var)

            print("Running compression 1/8 rank...")
            eigth_mean, eigth_var = TestNetwork.run_compress_iter(size, 1/8)
            run_eigth.append(eigth_mean)
            run_eigth_var.append(eigth_var)

            print("Running minimum weight selection...")
            mw_mean, mw_var = TestNetwork.run_min_weight(size)
            min_weight.append(mw_mean)
            min_weight_var.append(mw_var)

    print(stats.ttest_ind(min_weight, run_seven_eights)[1])
    print(stats.ttest_ind(min_weight, run_eigth)[1])

    print(TEST_SIZES)
    fig, ax1 = plt.subplots(1, 1)
    lines1 = plot_compression_results(
        (random, random_var), (run_seven_eights, run_full_var), (run_three_fourths, run_tf_var), (run_half, run_half_var), (run_fourth, run_fourth_var), (run_eigth, run_eigth_var), (min_weight, min_weight_var), ax1)
    # lines2 = plot_compression_results_var(
    #     random_var, run_full_var, run_tf_var, run_half_var, run_fourth_var, run_eigth_var, min_weight_var, ax2)
    fig.legend(lines1, labels=LABELS)
    plt.show()


if __name__ == '__main__':
    main()
