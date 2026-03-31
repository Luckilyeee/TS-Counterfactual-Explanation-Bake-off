from sklearn import preprocessing
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.utils import to_sklearn_dataset
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import warnings
import os
warnings.simplefilter("ignore", UserWarning)

def _resolve_ucr_root():
    env_root = os.environ.get("UCR_DATA_ROOT")
    candidates = [env_root] if env_root else []

    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        candidates.append(os.path.join(current, "UCRArchive_2018"))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    candidates.append("/UCRArchive_2018")
    for root in candidates:
        if root and os.path.isdir(root):
            return root
    raise FileNotFoundError(
        "Could not locate 'UCRArchive_2018'. Put it at repo root or set UCR_DATA_ROOT."
    )

def getmetrics(x1, x2):
    x1 = np.round(x1, 3)
    x2 = np.round(x2, 3)

    l = np.round(x1 - x2, 3)
    l1 = distance.cityblock(x1, x2)
    l2 = np.linalg.norm(x1 - x2)  # Correct usage of np.linalg.norm for one-dimensional arrays
    l_inf = distance.chebyshev(x1, x2)
    sparsity = (len(l) - np.count_nonzero(l)) / len(l)

    segnums = get_segmentsNumber(l)
    return l1, l2, l_inf, sparsity, segnums

def label_encoder(training_labels, testing_labels):

    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)

    return y_train, y_test

def read_data(ds_name):
    ucr_root = _resolve_ucr_root()
    train_data = np.loadtxt(os.path.join(ucr_root, ds_name, f"{ds_name}_TRAIN.tsv"), delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    # print(x_train.shape, y_train.shape)

    test_data = np.loadtxt(os.path.join(ucr_root, ds_name, f"{ds_name}_TEST.tsv"), delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    # print(x_test.shape, y_test.shape)

    y_train, y_test = label_encoder(y_train, y_test)

    return x_train, y_train, x_test, y_test

def get_segmentsNumber(l4):
    flag, count = 0,0
    for i in range(len(l4)):
        if l4[i:i+1][0]!=0:
            flag=1
        if flag==1 and l4[i:i+1][0]==0:
            count= count+1
            flag=0
    return count

def cf_ood(X_train, counterfactual_examples):

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(X_train))), novelty=True, metric='euclidean')
    lof.fit(to_sklearn_dataset(X_train))

    novelty_detection = lof.predict(to_sklearn_dataset(counterfactual_examples))

    ood= np.count_nonzero(novelty_detection == -1)
    OOD_lof = ood / len(counterfactual_examples)

    # One-Class SVM (OC-SVM)
    clf = OneClassSVM(gamma='scale', nu=0.02).fit(to_sklearn_dataset(X_train))

    novelty_detection = clf.predict(to_sklearn_dataset(counterfactual_examples))

    ood = np.count_nonzero(novelty_detection == -1)
    OOD_svm = ood/ len(counterfactual_examples)

    # Initialize a list to store OOD results for min_edit_cf
    OOD_ifo = []

    # Loop over different random seeds
    for seed in range(10):
        iforest = IsolationForest(random_state=seed).fit(to_sklearn_dataset(X_train))

        novelty_detection = iforest.predict(to_sklearn_dataset(counterfactual_examples))

        ood = np.count_nonzero(novelty_detection == -1)

        OOD_ifo.append((ood/ len(counterfactual_examples)))

    mean_OOD_ifo = np.mean(OOD_ifo)

    return mean_OOD_ifo, OOD_lof, OOD_svm

def plot(cam_swap_cf, X_test):
    plt.style.use("classic")
    colors = [
        '#08F7FE',  # teal/cyan
        '#FE53BB',  # pink
        '#F5D300',  # yellow
        '#00ff41',  # matrix green
    ]
    df = pd.DataFrame({'Predicted: ': list(X_test[10].flatten()),
                       'Counterfactual: ': list(cam_swap_cf[10].flatten())})
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(marker='.', color=colors, ax=ax)
    # Redraw the data with low alpha and slighty increased linewidth:
    n_shades = 10
    diff_linewidth = 1.05
    alpha_value = 0.3 / n_shades
    for n in range(1, n_shades + 1):
        df.plot(marker='.',
                linewidth=2 + (diff_linewidth * n),
                alpha=alpha_value,
                legend=False,
                ax=ax,
                color=colors)

    ax.grid(color='#2A3459')
    plt.xlabel('Time', fontweight='bold', fontsize='large')
    plt.ylabel('Value', fontweight='bold', fontsize='large')
    # plt.savefig('../Images/Initial_Example_Neon.pdf')
    plt.show()


def plot_cf_vs_ori(DS, index, counterfactual_examples, save_path=None):
    _, _, X_test, _ = read_data(DS)
    plt.style.use('bmh')
    # plt.plot(X_test[index], label='Original', color='magenta')
    plt.plot(counterfactual_examples[index], label='CF', ls='--', color='green')

    plt.xlabel('Time', fontsize=12)
    plt.title('CF vs Ori on index ' + str(index), fontsize=14)
    plt.legend()

    # Save the figure if a save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
