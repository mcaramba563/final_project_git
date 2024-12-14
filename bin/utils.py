import glob
import numpy as np
import os.path

def get_random_pathes(n):
    """
        Retrieves random image paths and their corresponding labels from the MNIST dataset

        :param n: Number of random image paths and labels to return.
        :type n: int

        :returns: A tuple containing a list of n shuffled image paths (list of str), a list of n corresponding labels (list of int).
        :rtype: tuple(list[str], list[int])
    """
    pathes = []
    labels = []
    for i in range(0, 10):
        cur_i_path = glob.glob(f'images/mnist_png/test/{i}/*.png')
        cur_labels = [i] * len(cur_i_path)
        pathes.extend(cur_i_path)
        labels.extend(cur_labels)

    permute = np.random.permutation(len(pathes))

    new_pathes = [''] * len(pathes)
    new_labels = [0] * len(pathes)
    for i in range(len(pathes)):
        new_pathes[i] = pathes[permute[i]]
        new_labels[i] = labels[permute[i]]
    return (new_pathes[:n], new_labels[:n])


def file_exists(file_path):
    """
        Checks whether a file exists at the given path

        :param file_path: The file path to check.
        :type file_path: str

        :returns: True if the file exists, False otherwise.
        :rtype: bool
    """
    return os.path.exists(file_path)