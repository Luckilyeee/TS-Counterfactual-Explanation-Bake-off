import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.convolution_based import RocketClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance

from pyts.transformation import ShapeletTransform
from tslearn.utils import to_sklearn_dataset

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


class NaiveShapelet:
    def __init__(self, dataset_name, classifier=None):
        self.dataset = dataset_name
        self.classifier = classifier or RocketClassifier(estimator=RandomForestClassifier(n_estimators=100), random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = self.read_data()

    def read_data(self):
        ucr_root = _resolve_ucr_root()
        train_data = np.loadtxt(os.path.join(ucr_root, self.dataset, f"{self.dataset}_TRAIN.tsv"), delimiter='\t')
        test_data = np.loadtxt(os.path.join(ucr_root, self.dataset, f"{self.dataset}_TEST.tsv"), delimiter='\t')

        x_train, y_train = train_data[:, 1:], train_data[:, 0]
        x_test, y_test = test_data[:, 1:], test_data[:, 0]

        le = preprocessing.LabelEncoder()
        le.fit(np.concatenate((y_train, y_test), axis=0))
        return x_train, x_test, le.transform(y_train), le.transform(y_test)

    def get_shapelets(self):
        len_ts = self.X_train.shape[1]
        window_sizes = [int(len_ts * r) for r in [0.3, 0.5, 0.7]]
        st = ShapeletTransform(n_shapelets=300, window_sizes=window_sizes, random_state=42, sort=True)
        st.fit_transform(self.X_train, self.y_train)
        return pd.DataFrame(st.indices_)

    def shapelet_category(self, indices, label):
        label_values = self.y_train[indices.iloc[:, 0]]
        res = pd.concat([indices, pd.DataFrame(label_values)], axis=1)
        res.columns = ["idx", "start", "end", "label"]
        return res.groupby('label').get_group(label).head(1).values

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier

    def eval_model(self, y_pred):
        return accuracy_score(self.y_test, y_pred)

    @staticmethod
    def target_(model, instance):
        target = np.argsort((model.predict_proba(instance.reshape(1, -1))))[0][-2:-1][0]
        return target

    def counterfactual_generation(self, shapelets, targets):
        X_test_copy = self.X_test.copy()
        counterfactuals = []
        for i, target in enumerate(targets):
            idx, start, end, _ = shapelets[target][0]
            cf = X_test_copy[i].copy()
            cf[start:end] = self.X_train[idx][start:end]
            counterfactuals.append(cf)
        return np.array(counterfactuals)

    def getmetrics(self, x1, x2):
        diff = np.round(x1 - x2, 3)
        l1 = distance.cityblock(x1, x2)
        l2 = np.linalg.norm(diff)
        l_inf = distance.chebyshev(x1, x2)
        sparsity = (len(diff) - np.count_nonzero(diff)) / len(diff)
        segments = self.count_segments(diff)
        return l1, l2, l_inf, sparsity, segments

    def count_segments(self, diff):
        flag, count = 0, 0
        for val in diff:
            if val != 0:
                flag = 1
            elif flag:
                count += 1
                flag = 0
        return count

    def target_probability(self, counter_probs, targets):
        target_probs = [probs[t] for probs, t in zip(counter_probs, targets)]
        flips = [int(np.argmax(probs) == t) for probs, t in zip(counter_probs, targets)]
        return target_probs, sum(flips) / len(flips), flips

    def cf_ood(self, cfs):
        X_train = to_sklearn_dataset(self.X_train)
        cfs = to_sklearn_dataset(cfs)

        lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(X_train))), novelty=True).fit(X_train)
        OOD_lof = np.count_nonzero(lof.predict(cfs) == -1) / len(cfs)

        svm = OneClassSVM(gamma='scale', nu=0.02).fit(X_train)
        OOD_svm = np.count_nonzero(svm.predict(cfs) == -1) / len(cfs)

        OOD_ifo = [
            np.count_nonzero(IsolationForest(random_state=seed).fit(X_train).predict(cfs) == -1) / len(cfs)
            for seed in range(10)
        ]
        return OOD_svm, OOD_lof, np.mean(OOD_ifo)

    def run(self):
        shapelet_indices = self.get_shapelets()
        classes = np.unique(self.y_test)
        shapelets = [self.shapelet_category(shapelet_indices, c) for c in classes]

        model = self.train()
        accuracy = self.eval_model(model.predict(self.X_test))

        targets = [self.target_(model, x) for x in self.X_test]
        cfs = self.counterfactual_generation(shapelets, targets)
        counter_probs = model.predict_proba(cfs)
        target_probs, flip_rate, flips = self.target_probability(counter_probs, targets)

        results, valid_cfs = [], []
        for i, flipped in enumerate(flips):
            if flipped:
                metrics = self.getmetrics(cfs[i], self.X_test[i]) + (target_probs[i],)
                results.append(metrics)
                valid_cfs.append(cfs[i])
            else:
                valid_cfs.append(np.full_like(self.X_test[i], -1))

        # Save necessary files
        save_dir = os.path.join("res_rk", self.dataset)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "cfs.npy"), cfs)
        np.save(os.path.join(save_dir, "valid_cfs.npy"), np.array(valid_cfs))
        np.save(os.path.join(save_dir, "target_probs.npy"), np.array(target_probs))
        np.save(os.path.join(save_dir, "flips.npy"), np.array(flips))

        if results:
            results = np.array(results)
            stats = {
                'Dataset': self.dataset,
                'L1_mean': results[:, 0].mean(), 'L1_std': results[:, 0].std(),
                'L2_mean': results[:, 1].mean(), 'L2_std': results[:, 1].std(),
                'L_inf_mean': results[:, 2].mean(), 'L_inf_std': results[:, 2].std(),
                'Sparsity_mean': results[:, 3].mean(), 'Sparsity_std': results[:, 3].std(),
                'Segment_mean': results[:, 4].mean(), 'Segment_std': results[:, 4].std(),
                'target_prob_mean': results[:, 5].mean(), 'target_prob_std': results[:, 5].std(),
                'Flip_Label_Rate': flip_rate,
                'Model_Accuracy': accuracy,
            }
            OOD_svm, OOD_lof, OOD_ifo = self.cf_ood([cf for cf in valid_cfs if not np.all(cf == -1)])
            stats.update({'OOD_SVM': OOD_svm, 'OOD_LOF': OOD_lof, 'OOD_IFO': OOD_ifo})
            return stats
        return None

if __name__ == "__main__":
    datasets = [
    "ArrowHead", "BeetleFly", "BirdChicken", "Coffee", "ECG200", "FaceFour", "GunPoint",
    "Lightning2", "Plane", "Trace", "Chinatown", "CBF", "TwoLeadECG",
    "Beef", "Car", "Lightning7", "Computers", "OSULeaf",
    "Worms", "SwedishLeaf"
   ]
    results_path = "res_rk"
    os.makedirs(results_path, exist_ok=True)
    csv_path = os.path.join(results_path, "MG-RF.csv")

    for dataset in datasets:
        print(f"Running on {dataset}")
        runner = NaiveShapelet(dataset)
        stats = runner.run()
        if stats:
            df = pd.DataFrame([stats])
            df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        print(f"Finished {dataset}")

# nohup python main.py > output_rk_ALL.log 2>&1 &
