import numpy as np
import os
from tensorflow import keras
import pandas as pd
import time
import classifiers.fcn_val as fcn
from tslearn.neighbors import KNeighborsTimeSeries

from utils import cf_ood, getmetrics, read_data

def load_fcn(
    x_train,
    y_train,
    x_test,
    y_test,
    output_directory=None,
    nb_epochs=1500,
    weights_directory=None,
):
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = fcn.Classifier_FCN(
        output_directory,
        input_shape,
        nb_classes,
        verbose=True,
        load_weights=True,
        weights_directory=weights_directory,
    )

    return classifier.model

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


def findSubarray(a, k):
    n = len(a)
    if k > n or k <= 0:
        raise ValueError(f"Invalid subarray length k={k} for array of length {n}")
    vec = []
    for i in range(n - k + 1):
        temp = []
        for j in range(i, i + k):
            temp.append(a[j])
        vec.append(temp)
    sum_arr = [np.sum(v) for v in vec]
    return np.array(vec[np.argmax(sum_arr)])

def predict(X, model):
    if len(X.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        X = X.reshape((X.shape[0], X.shape[1], 1))
    elif len(X.shape) == 1:
        X = X.reshape((1, -1, 1))

    y_pred = model.predict(X, verbose=0)

    return np.argmax(y_pred, axis=1)

def target_(query):
    target = np.argsort((model.predict(query.reshape(1, -1, 1), verbose=0)))[0][-2:-1][0]
    return target

def counterfactual_generator_swap(instance, nun, subarray_length):
    target = target_(X_test[instance])
    flip = False
    most_influencial_array = findSubarray((training_weights[nun]), subarray_length)

    starting_point = np.where(training_weights[nun] == most_influencial_array[0])[0][0]

    X_example = np.concatenate((X_test[instance][:starting_point],
                                (X_train[nun][starting_point:subarray_length + starting_point]),
                                X_test[instance][subarray_length + starting_point:]))

    prob_target = model.predict(X_example.reshape(1, -1, 1), verbose=0)[0][y_pred[instance]]

    max_length = len(training_weights[nun])
    while prob_target > 0.5:
        subarray_length += 1
        if subarray_length > max_length:
            # Stop to avoid invalid subarray length
            break

        most_influencial_array = findSubarray((training_weights[nun]), subarray_length)
        starting_point = np.where(training_weights[nun] == most_influencial_array[0])[0][0]
        X_example = np.concatenate((X_test[instance][:starting_point],
                                    (X_train[nun][starting_point:subarray_length + starting_point]),
                                    X_test[instance][subarray_length + starting_point:]))
        prob_target = model.predict(X_example.reshape(1, -1, 1), verbose=0)[0][y_pred[instance]]

    cf_proba = model.predict(X_example.reshape(1, -1, 1), verbose=0)[0][target]
    cf_pred = np.argmax(model.predict(X_example.reshape(1, -1, 1), verbose=0), axis=1)[0]
    if target == cf_pred:
        flip = True
    return X_example, cf_proba, flip

DATASETS_LIST = [
    "BeetleFly", "BirdChicken", "Coffee", "ECG200", "FaceFour", "GunPoint",
    "Lightning2", "Plane", "Trace", "Chinatown", "CBF", "TwoLeadECG",
    "Beef", "Car", "ArrowHead", "Lightning7", "Computers", "OSULeaf",
    "Worms", "SwedishLeaf"
]

for dataset in DATASETS_LIST:
    print("start ", dataset)
    start_time = time.time()
    X_train, y_train, X_test, y_test = read_data(str(dataset))
    classes = np.unique(y_train)
    nb_classes = len(classes)
    # y_train = keras.utils.to_categorical(y_train, num_classes=nb_classes)


    model_path1 = '/UCR/'  + dataset + "/models" # the path to the saved fcn model weights
    model = load_fcn(X_train, y_train, X_test, y_test,
                         output_directory='', weights_directory=model_path1)

    y_test = keras.utils.to_categorical(y_test, num_classes=nb_classes)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy: ', score[1])

    y_pred = predict(X_test, model)
    path = '...' # the path to the saved train weights
    training_weights = np.load(path + str(dataset) + '/cam_train_weights.npy')

    nuns = []
    n = len(X_test)
    for instance in range(n):
        nuns.append(native_guide_retrieval(X_test[instance], 'euclidean', 1)[1][0])

    nuns = np.array(nuns)
    test_instances = np.array(range(n))

    cf_cam_swap = []
    target_probs = []
    flips0 = []
    for test_instance, nun in zip(test_instances, nuns):
        cf, target_prob, flip0 = counterfactual_generator_swap(test_instance, nun, 1)
        cf_cam_swap.append(cf)
        target_probs.append(target_prob)
        flips0.append(flip0)
    cf_cam_swap = np.array(cf_cam_swap)
    cf_cam_swap = cf_cam_swap.reshape(cf_cam_swap.shape[0], cf_cam_swap.shape[1])
    np.save('/NG/src/results_cam/' + str(dataset) + '_cam_cfori.npy', cf_cam_swap)
    runtime = time.time() - start_time

    cfs = []

    l1s = []
    l2s = []
    l_infs = []
    Sparsitys = []
    segnums = []
    Targets = []
    Targets_all = []
    flips = []

    for i in range(len(cf_cam_swap)):
        cf = cf_cam_swap[i]
        query = X_test[i]
        target_probability = target_probs[i]
        Targets_all.append(target_probability)
        if flips0[i] :
            cfs.append(cf)
            flips.append(1)
            l1, l2, l_inf, sparsity, segnum = getmetrics(cf.flatten(), query.flatten())
            l1s.append(l1)
            l2s.append(l2)
            l_infs.append(l_inf)
            segnums.append(segnum)
            Sparsitys.append(sparsity)
            Targets.append(target_probability)
        else:
            flips.append(0)
            cfs.append(np.full_like(cf, fill_value=-1))

    L1s_arr = np.array(l1s)
    L2s_arr = np.array(l2s)
    L_infs_arr = np.array(l_infs)
    Sparsitys_arr = np.array(Sparsitys)
    Segments_arr = np.array(segnums)
    target_probs_arr = np.array(Targets)
    target_probs_all = np.array(Targets_all)

    cfs = np.array(cfs)
    L1_mean, L1_std = L1s_arr.mean(), L1s_arr.std()
    L2_mean, L2_std = L2s_arr.mean(), L2s_arr.std()
    L_inf_mean, L_inf_std = L_infs_arr.mean(), L_infs_arr.std()
    Sparsity_mean, sparsity_std = Sparsitys_arr.mean(), Sparsitys_arr.std()
    Segment_mean, segment_std = Segments_arr.mean(), Segments_arr.std()
    target_prob_mean, target_prob_std = target_probs_arr.mean(), target_probs_arr.std()
    valid_cfs = [cf for cf in cfs if not np.all(cf == -1)]
    mean_OOD_ifo, OOD_lof, OOD_svm = cf_ood(valid_cfs, X_train)
    flip_rate = sum(flips) / len(flips)

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
        'flip_rate': flip_rate,
        'mean_OOD_ifo': mean_OOD_ifo,
        'OOD_lof': OOD_lof,
        'OOD_svm': OOD_svm,
        'runtime': runtime/len(X_test),
        'accuracy': score[1]
    }

    # Create a DataFrame from the dictionary
    stats_dict['Dataset'] = dataset
    # Save results immediately after each dataset
    stats_df = pd.DataFrame([stats_dict])
    results_csv_path = '/NG/src/results_cam/all_datasets_cam.csv'

    if not os.path.exists(results_csv_path):
        stats_df.to_csv(results_csv_path, index=False)
    else:
        stats_df.to_csv(results_csv_path, mode='a', header=False, index=False)

    np.save('/NG/src/results_cam/' + dataset + '_cam_cfs.npy', cfs)
    np.save('/NG/src/results_cam/' + dataset + '_cam_target.npy', target_probs_all)
    np.save('/NG/src/results_cam/' + dataset + '_cam_flips.npy', flips)
    print("finished ", dataset)

print("✅ All dataset results saved to all_datasets_cam.csv")



# nohup python3 cam.py > cam_new.log 2>&1 &
# conda activate sg
