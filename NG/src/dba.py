import numpy as np
import os
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.convolution_based import RocketClassifier
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tslearn.barycenters import dtw_barycenter_averaging
import warnings
warnings.simplefilter("ignore", UserWarning)
from utils import (cf_ood, getmetrics)

def _resolve_ucr_root():
    env_root = os.environ.get("UCR_DATA_ROOT")
    candidates = [env_root] if env_root else []

    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
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

def read_data(ds_name):
    ucr_root = _resolve_ucr_root()
    train_data = np.loadtxt(os.path.join(ucr_root, ds_name, f"{ds_name}_TRAIN.tsv"), delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    print(x_train.shape, y_train.shape)

    test_data = np.loadtxt(os.path.join(ucr_root, ds_name, f"{ds_name}_TEST.tsv"), delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    print(x_test.shape, y_test.shape)

    y_train, y_test = label_encoder(y_train, y_test)

    return x_train, y_train, x_test, y_test

def label_encoder(training_labels, testing_labels):
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)

    return y_train, y_test

def target_(instance):
    target = np.argsort((model.predict_proba(instance.reshape(1,-1))))[0][-2:-1][0]
    return target

def native_guide_retrieval(query, distance, n_neighbors):
    alt_class = target_(query)  # second highest probability class

    df = pd.DataFrame(y_train, columns=['label'])
    df.index.name = 'index'

    ts_length = X_train.shape[1]

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)

    # Fit only on samples with the alternative class label
    knn.fit(X_train[list(df[df['label'] == alt_class].index.values)])

    dist, ind = knn.kneighbors(query.reshape(1, ts_length), return_distance=True)

    # Return distances and original indices of those neighbors
    return dist[0], df[df['label'] == alt_class].index[ind[0][:]]

datasets_list = [
    "BeetleFly", "BirdChicken", "Coffee", "ECG200", "FaceFour", "GunPoint",
    "Lightning2", "Plane", "Trace", "Chinatown", "CBF", "TwoLeadECG",
    "Beef", "Car", "ArrowHead", "Lightning7", "Computers", "OSULeaf",
    "Worms", "SwedishLeaf"
]

all_results = []  # list to hold stats_dicts for all datasets


for DS in datasets_list: # 'Coffee'
    print("start ", DS)
    X_train, y_train, X_test, y_test = read_data(DS)
    # fit the classifier
    model = RocketClassifier(estimator=RandomForestClassifier(n_estimators=100), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy: ", score)

    res_data = []  # Initialize a list to store dictionaries of data
    cfs = []

    l1s = []
    l2s = []
    l_infs = []
    Sparsitys = []
    segnums = []
    flip_rate = 0
    Targets = []
    Targets_all = []
    flips = []
    defaults = []
    n = len(X_test)
    for i in range(n):
        # print("working on ", i)
        beta = 0
        pred_treshold = 0.5
        query = X_test[i]
        dis, idx = native_guide_retrieval(X_test[i], 'dtw', 1)
        insample_cf = X_train[idx.item()]
        target = target_(query)

        generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1 - beta), beta]))

        prob_target = model.predict_proba(generated_cf.reshape(1, -1))[0][target]

        while prob_target <= pred_treshold:
            if beta >= 1:
                # print('defaulting')
                defaults.append(1)
                generated_cf = insample_cf
                prob_target = model.predict_proba(generated_cf.reshape(1, -1))[0][target]
                break
            else:
                defaults.append(0)
                beta += 0.05
                generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1 - beta), beta]))
                prob_target = model.predict_proba(generated_cf.reshape(1, -1))[0][target]

        predicted_class = model.predict(generated_cf.reshape(1, -1))[0]
        # print(predicted_class, target)

        Targets_all.append(prob_target)
        generated_cf = generated_cf.flatten()
        l1, l2, l_inf, sparsity, segnum = getmetrics(generated_cf, query.flatten())

        if predicted_class == target:
            flip_rate+=1
            l1s.append(l1)
            l2s.append(l2)
            l_infs.append(l_inf)
            Sparsitys.append(sparsity)
            segnums.append(segnum)
            Targets.append(prob_target)
            flips.append(1)
            cfs.append(generated_cf)
        else:
            flips.append(0)
            cfs.append(np.full_like(generated_cf, fill_value=-1))
    L1s_arr = np.array(l1s)
    L2s_arr = np.array(l2s)
    L_infs_arr = np.array(l_infs)
    Sparsitys_arr = np.array(Sparsitys)
    target_probs_arr = np.array(Targets)
    target_probs_all = np.array(Targets_all)
    defaults = np.array(defaults)
    Segment_arr = np.array(segnums)
    cfs = np.array(cfs)
    # print(cfs.shape)
    L1_mean, L1_std = L1s_arr.mean(), L1s_arr.std()
    L2_mean, L2_std = L2s_arr.mean(), L2s_arr.std()
    L_inf_mean, L_inf_std = L_infs_arr.mean(), L_infs_arr.std()
    Sparsity_mean, sparsity_std = Sparsitys_arr.mean(), Sparsitys_arr.std()
    Segment_mean, segment_std = Segment_arr.mean(), Segment_arr.std()
    target_prob_mean, target_prob_std = target_probs_arr.mean(), target_probs_arr.std()
    flip = flip_rate / (len(flips))
    valid_cfs = [cf for cf in cfs if not np.all(cf == -1)]
    mean_OOD_ifo, OOD_lof, OOD_svm = cf_ood(valid_cfs, X_train)

    # Collect the statistics into a dictionary
    stats_dict = {
        'L1_mean': L1_mean,
        'L1_std': L1_std,
        'L2_mean': L2_mean,
        'L2_std': L2_std,
        'L_inf_mean': L_inf_mean,
        'L_inf_std': L_inf_std,
        'Sparsity_mean': Sparsity_mean,
        'Sparsity_std': sparsity_std,
        'Segment_mean': Segment_mean,
        'Segment_std': segment_std,
        'target_prob_mean': target_prob_mean,
        'target_prob_std': target_prob_std,
        'flip_rate': flip,
        'mean_OOD_ifo': mean_OOD_ifo,
        'OOD_lof': OOD_lof,
        'OOD_svm': OOD_svm,
        'accuracy': score
    }

    stats_dict['Dataset'] = DS
    # Convert to DataFrame
    stats_df = pd.DataFrame([stats_dict])

    # Path to final results CSV
    csv_path = '/NG/src/results_dba_rocket/NG-DBA.csv'

    # Append to CSV (write header only if file doesn't exist)
    stats_df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    np.save('/NG/src/results_dba_rocket/' + str(DS) + '_cfs.npy', cfs)
    np.save('/NG/src/results_dba_rocket/' + str(DS) + '_target_probs.npy', target_probs_arr)
    np.save('/NG/src/results_dba_rocket/' + str(DS) + '_target_probs_all.npy', target_probs_all)
    np.save('/NG/src/results_dba_rocket/' + str(DS) + '_flips.npy', flips)
    np.save('/NG/src/results_dba_rocket/' + str(DS) + '_defaults.npy', defaults)
    print("finished ", DS)

print("✅ All dataset results saved to NG-DBA.csv")


# nohup python3 dba.py > dba_new.log 2>&1 &
# conda activate aeon-env



