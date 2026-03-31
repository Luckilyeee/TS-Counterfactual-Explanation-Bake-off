import numpy as np
import pandas as pd
from utils import read_data, getmetrics, cf_ood

datasets = ["Coffee", "GunPoint", "ECG200", "Chinatown", "BirdChicken", "BeetleFly", "Lightning2", "FaceFour"]
path = "..." #path to the saved results
for dataset in datasets:
    l1s = []
    l2s = []
    l_infs = []
    Sparsitys = []
    segnums = []
    Targets = []
    Targets_all = []
    flips = []
    print("start ", dataset)
    X_train, y_train, X_test, y_test = read_data(str(dataset))
    cfs_all = np.load(path + dataset + '/' + 'saliency_cf.npy')
    cf_probs_all = np.load(path + dataset + '/' + 'saliency_cf_prob.npy')
    for i in range(len(cfs_all)):
        Targets_all.append(cf_probs_all[i])
        if cf_probs_all[i] > 0.5:
            flips.append(1)
            l1, l2, l_inf, sparsity, segnum = getmetrics(cfs_all[i].flatten(), X_test[i].flatten())
            l1s.append(l1)
            l2s.append(l2)
            l_infs.append(l_inf)
            segnums.append(segnum)
            Sparsitys.append(sparsity)
            Targets.append(cf_probs_all[i])
        else:
            flips.append(0)
            cfs_all[i] = np.full_like(cfs_all[i], fill_value=-1)
    L1s_arr = np.array(l1s)
    L2s_arr = np.array(l2s)
    L_infs_arr = np.array(l_infs)
    Sparsitys_arr = np.array(Sparsitys)
    Segments_arr = np.array(segnums)
    target_probs_arr = np.array(Targets)
    target_probs_all = np.array(Targets_all)

    cfs = np.array(cfs_all)
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
    }

    # Create a DataFrame from the dictionary
    stats_df = pd.DataFrame([stats_dict])
    stats_df.to_csv(path + dataset + '/' + 'cels_eva_res.csv', index=False)
    np.save(path + dataset + '/' + 'cels_finalcf.npy', cfs)
    np.save(path + dataset + '/' + 'cels_target.npy', target_probs_all)
    np.save(path + dataset + '/' + 'cels_flips.npy', flips)

    targets_data = np.load(path + dataset + '/' + 'cels_target.npy')
    flips_data = np.load(path + dataset + '/' + 'cels_flips.npy')
    print(targets_data)
    print(flips_data)
    print("finished ", dataset)

