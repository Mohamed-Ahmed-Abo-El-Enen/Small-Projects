import matplotlib.pyplot as plt
from utils.mnist_read import reconstruct_x_dataset_img
import os


def visualize_reconstructed_img(sub_root, g_plot, itr=-1, save_fig=False):
    num_row = 2
    num_col = 5

    g_plot = reconstruct_x_dataset_img(g_plot)

    fig, ax = plt.subplots(num_row, num_col)
    i = 0

    for row in range(num_row):
        for col in range(num_col):
            plt.sca(ax[row, col])
            plt.imshow(g_plot[i], cmap='gray')
            plt.axis('off')
            i += 1

    plt.title('Samples at Iteration %d' % itr, loc='left')
    if save_fig:
        plt.savefig(os.path.join(sub_root,'iteration_%d.png' % itr))
        plt.close()
    else:
        plt.show()