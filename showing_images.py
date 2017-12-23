import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64

###############################################################################
# This code is mostly taken from
# https://github.com/andrwc/Adversarial-MNIST
# but there have been little changes.
# The original file that this code is copied from was analysis.py
###############################################################################

"""General analysis and some viz"""


def get_adversarial_sixes(orig, adver, idxes):
    """(np.array, np.array, list) -> (np.array, np.array)
    Return a tuple containing the original mnist and adversarial mnist images,
    respectively, that were found to be classified as the digit six with high
    confidence

    :param orig: the original mnist images for the digit 2.
    :param adver: the adversarial mnist images for the digit 2.
    """
    return orig[idxes], adver[idxes]


if __name__ == '__main__':

    origtwos = np.load('original_twos.npy')
    adtwos = np.load('adversarial_twos.npy')
    adtwos_pred = np.load('adversarial_twos_pred.npy')

    preds = pd.DataFrame(adtwos_pred)

    predclasses = pd.DataFrame(np.argmax(adtwos_pred, axis=1))
    sixes_idx = predclasses[predclasses[0] == 6].index

    predscore = pd.Series(np.max(adtwos_pred[sixes_idx], axis=1))
    # predscore.sort(ascending=False)  # inplace

    orig, adv = get_adversarial_sixes(origtwos, adtwos, sixes_idx)

    #######################################
    # This part is added by myself
    #######################################

    fig = plt.figure()
    for i in predscore.index[:10]:
        a = fig.add_subplot(10, 3, i*3+1)
        # imgplot = plt.imshow(orig[i].reshape(28,28))
        a.matshow(orig[i].reshape(28,28), cmap=plt.cm.binary)
        plt.xticks(())
        plt.yticks(())
        a = fig.add_subplot(10, 3, i*3+2)
        # imgplot = plt.imshow((adv[i] - orig[i]).reshape(28, 28))
        a.matshow((adv[i] - orig[i]).reshape(28, 28), cmap=plt.cm.binary)
        plt.xticks(())
        plt.yticks(())
        a = fig.add_subplot(10, 3, i*3+3)
        # imgplot = plt.imshow(adv[i].reshape(28, 28))
        a.matshow(adv[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xticks(())
        plt.yticks(())

    plt.savefig("image.png")