import tensorflow as tf
import pyts.datasets
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
                BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from alibi.explainers import tfcounterfactual, Counterfactual
import pandas as pd
from sklearn.ensemble import IsolationForest
from tslearn.barycenters import dtw_barycenter_averaging
import pyts.datasets
from pyts.transformation import ShapeletTransform
import sklearn
import warnings
from scipy.spatial import distance
import logging, fileHandlerWithHeader
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

import numpy as np

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

def readUCR(ds_name):
    path = _resolve_ucr_root()
    train_data = np.loadtxt(os.path.join(path, ds_name, f"{ds_name}_TRAIN.tsv"), delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    # print(x_train.shape, y_train.shape)

    test_data = np.loadtxt(os.path.join(path, ds_name, f"{ds_name}_TEST.tsv"), delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    # print(x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test

def getmetrics(x1,x2):
    x1 = [np.round(e, 3) for e in x1]
    x2 = [np.round(e, 3) for e in x2]

    l = [np.round(e1-e2,3) for e1,e2 in zip(x1,x2)]
    dist = distance.cityblock(x1,x2)
    sparsity = (len(l)-np.count_nonzero(l))/len(l)

    segnums = get_segmentsNumber(l)
    return dist, sparsity, segnums

def get_segmentsNumber(l4):
    flag, count = 0,0
    for i in range(len(l4)):
        if l4[i:i+1][0]!=0:
            flag=1
        if flag==1 and l4[i:i+1][0]==0:
            count= count+1
            flag=0
    return count

def get_shapelet(X_train, y_train, len_ts):
    a = int(len_ts * 0.3)
    b = int(len_ts * 0.5)
    c = int(len_ts * 0.7)
    st = ShapeletTransform(n_shapelets=300, window_sizes=[a, b, c],
                           random_state=42, sort=True)
    st.fit_transform(X_train, y_train)
    indices = pd.DataFrame(st.indices_)
    return indices

# category the shapelets by their label, res include 4 columns, the index, start index, end index, and the label
def shapelet_category(y_train, idx_shapelets):
    idx_shapelets.columns = ['index', 'start_point', 'end_point']
    idx_shapelets = idx_shapelets.astype({'index': 'int', 'start_point': 'int', 'end_point': 'int'})
    idx_shapelets['class'] = y_train[idx_shapelets['index']].astype(int)
    selected_shapelets = idx_shapelets.drop_duplicates(subset='class')
    selected_shapelets = selected_shapelets.sort_values(by='class')
    print(selected_shapelets)

    return selected_shapelets

for DS in ['SwedishLeaf']:
    lgr = logging.getLogger('LogDS')
    lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
    # add a file handler #     # ACF_distProto
    fh = fileHandlerWithHeader.FileHandlerWithHeader("./logs/SG_CF/"+DS + '_LOG_.csv',
                                                     header='L1_dist,sparsity_per,num_segments,runtime,Target_class_prob,flip')
    fh.setLevel(logging.DEBUG)  # ensure all messages are logged to file

    frmt = logging.Formatter('%(message)s')
    fh.setFormatter(frmt)
    #
    # # add the Handler to the logger
    lgr.addHandler(fh)

    print("Loaded Dataset.."+str(DS))
    xtrain, ytrain, xtest, ytest_ = readUCR(DS)
    classes =np.unique(ytest_)
    nb_classes = len(classes)

    len_ts = xtrain.shape[1]
    idx_shapelets = get_shapelet(xtrain, ytrain, len_ts)  # index, start_point, end_point
    selected_shapelets = shapelet_category(ytrain, idx_shapelets)
    shapelet_dic = {}
    for cls in range(nb_classes):
        shapelet_dic[cls] = xtrain[selected_shapelets.iloc[cls]['index']][
                            selected_shapelets.iloc[cls]['start_point']:selected_shapelets.iloc[cls]['end_point']]

    print("Number of Classes "+str(nb_classes))
    proto_dic = {}
    protoptypes_lst = []

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((ytrain, ytest_), axis=0).reshape(-1, 1))
    ytrain = enc.transform(ytrain.reshape(-1, 1)).toarray()
    ytest = enc.transform(ytest_.reshape(-1, 1)).toarray()

    for cls in range(len(classes)):

        proto_dic[cls] = xtrain[selected_shapelets.iloc[cls]['index']]
        print(proto_dic[cls].shape)
        plt.plot(proto_dic[cls], linestyle="--",color='green', linewidth=2)
        plt.plot(np.arange(selected_shapelets.iloc[cls]['start_point'], selected_shapelets.iloc[cls]['end_point']),
                 xtrain[selected_shapelets.iloc[cls]['index'], selected_shapelets.iloc[cls]['start_point']:selected_shapelets.iloc[cls]['end_point']],
                 lw=3, color='red')
        plt.title("Proto of class #: "+str(cls))
        plt.grid()
        plt.show()

    # Transform input array into tensor
    xtrain = np.reshape(np.array(xtrain),(xtrain.shape[0], xtrain.shape[1],1))
    xtest = np.reshape(np.array(xtest),(xtest.shape[0], xtest.shape[1],1))


    path = './fcn_weights/'
    cnn = load_model(path + DS + '_best_model.hdf5')  # './fcn_weights/'
    score = cnn.evaluate(xtest, ytest, verbose=0)
    print('Test accuracy: ', score[1])

    cfs = []

    for instance in range(xtest.shape[0]):   #xtest.shape[0]
        X = xtest[instance].reshape((1,) + xtest[0].shape)
        shape = (1,) + xtrain.shape[1:]
        target_proba = 1
        tol = 0.01 # want counterfactuals with p(class)>0.99
        target_class = 'other' # any class other than 7 will do
        max_iter = 500
        lam_init = 1e-1
        max_lam_steps = 1
        learning_rate_init = 0.1
        feature_range = (xtrain.min(),xtrain.max())

        otherclasses_lst = list(shapelet_dic.keys())
        otherclasses_lst.remove(list(ytest[instance:instance+1][0]).index(1))
        print("Test sample class is "+str(list(ytest[instance:instance+1][0]).index(1)))

        protoDistance_dic ={}
        for c in otherclasses_lst:
            tmp = sum(abs(val1-val2) for val1, val2 in zip(xtest[instance:instance + 1].reshape(1,len(proto_dic[c]))[0], proto_dic[c]))
            protoDistance_dic[c] = tmp

        otherclasses_lst =[]
        for i in range(nb_classes-1):
            minval = min(protoDistance_dic.values())
            res = [k for k, v in protoDistance_dic.items() if v == minval][0]
            del protoDistance_dic[res]

            otherclasses_lst.append(res)

        print("sorted list is "+str(otherclasses_lst))

        # Start CF by closet CF onward
        for i, c  in enumerate(otherclasses_lst[:1]):
            print("choice"+str(i))
            start_time = time.time()
            len_shape1 = len(shapelet_dic[c])  # this is to assign the lengths of different shapelets
            # len_shape2 = len(shapelet_dic[c])
            shapelet1 = xtrain[selected_shapelets.iloc[c]['index']][selected_shapelets.iloc[c]['start_point']:selected_shapelets.iloc[c]['end_point']]
            shapelet1 = shapelet1.reshape((1, len_shape1, 1))
            # shapelet2 = shapelet_dic[c].reshape((1, len_shape2, 1))
            start_idx = selected_shapelets.iloc[c]['start_point']
            end_idx = selected_shapelets.iloc[c]['end_point']
            print("Explanation using prototype class "+str(c))
            cf = tfcounterfactual.TFCounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,  #outlier_model=outlier_model,
                                                   target_class=c, target_classid = c,max_iter=max_iter, lam_init=lam_init,
                                                   max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                                   feature_range=feature_range, tstID=i, dataset=DS, num_classes=nb_classes, choice=i,
                                                   len_shapelet1=len_shape1,
                                                   shapelet1=shapelet1,
                                                   start_idx = start_idx, end_idx = end_idx) #target_class

            explanation = cf.explain(X, proto_dic[c].reshape((1,) + xtest[0].shape))
            end = time.time() - start_time
            print("Execution Time is " + str(end))
            a = explanation.cf
            b = np.reshape(X, (1, X.shape[1]))[0]
            dist, sparsity, segnums = getmetrics(a,b)

            cfs.append(list(a))

            tmp = [dist,sparsity,segnums,end,explanation.Target_class_prob, explanation.flip]
            lgr.info(",".join(repr(e) for e in tmp))

    cfs = np.save("./logs/SG_CF/"+DS + '_cfs.npy', cfs)
    lgr.removeHandler(fh)
    fh.close()
    tf.keras.backend.clear_session()
