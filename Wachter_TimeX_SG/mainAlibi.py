import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import load_model

from sklearn import preprocessing
import numpy as np
import time
from alibi.explainers import Counterfactual
import sklearn
import warnings
from scipy.spatial import distance
import logging, fileHandlerWithHeader
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import os

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
    ucr_root = _resolve_ucr_root()
    train_data = np.loadtxt(os.path.join(ucr_root, ds_name, f"{ds_name}_TRAIN.tsv"), delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    # print(x_train.shape, y_train.shape)

    test_data = np.loadtxt(os.path.join(ucr_root, ds_name, f"{ds_name}_TEST.tsv"), delimiter='\t')
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
            flag=1
        if flag==1 and l4[i:i+1][0]==0:
            count= count+1
            flag=0
    return count

def target_(instance, model):
    #Let's Make CF class the second most probable class according to original prediction
    target = np.argsort((model.predict(instance.reshape(1,-1,1))))[0][-2:-1][0]
    return target

for name in ['Coffee']:
    # create logger
    lgr = logging.getLogger('LogDS')
    lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
    # add a file handler
    fh = fileHandlerWithHeader.FileHandlerWithHeader("./logs/ALIBI/"+ name + '_res.csv',
                                                     header='L1_dist,sparsity_per,num_segments,runtime,target_proba,flip')
    fh.setLevel(logging.DEBUG)  # ensure all messages are logged to file


    frmt = logging.Formatter('%(message)s')
    fh.setFormatter(frmt)
    #
    # # add the Handler to the logger
    lgr.addHandler(fh)
    print("Loaded Dataset.."+str(name))

    xtrain, ytrain, xtest, ytest_ = readUCR(name)
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest_.shape)

    import sklearn.preprocessing
    # print(xtest)
    classes = np.unique(ytrain)
    nb_classes = len(classes)
    print("Number of Classes " + str(nb_classes))

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((ytrain, ytest_), axis=0).reshape(-1, 1))
    ytrain = enc.transform(ytrain.reshape(-1, 1)).toarray()
    ytest = enc.transform(ytest_.reshape(-1, 1)).toarray()

    xtrain = np.reshape(np.array(xtrain), (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = np.reshape(np.array(xtest), (xtest.shape[0], xtest.shape[1], 1))

    path = './fcn_weights/'
    cnn = load_model(path + name + '_best_model.hdf5')
    score = cnn.evaluate(xtest, ytest, verbose=0)
    print('Test accuracy: ', score[1])

    cfs = []

    for instance in range(xtest.shape[0]):
        print("start working on instance ", instance)
        X = xtest[instance].reshape((1,) + xtest[0].shape)
        print(X.shape)
        target = target_(X, cnn)
        shape = (1,) + xtrain.shape[1:]
        target_proba = 1
        tol = 0.01 # want counterfactuals with p(class)>0.99
        target_class = target # any class other than 7 will do
        max_iter = 1000
        lam_init = 1e-1
        max_lam_steps = 1
        learning_rate_init = 0.1
        feature_range = (xtrain.min(),xtrain.max())

        start_time = time.time()
        cf = Counterfactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                            target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                            feature_range=feature_range)
        explanation = cf.explain(X)
        end = time.time() - start_time
        print("Execution Time is " + str(end))

        a = explanation.cf
        cfs.append(list(a))
        b = np.reshape(X, (1, X.shape[1]*X.shape[2]))[0]

        dist, sparsity, segnums = getmetrics(a,b)

        tmp = [dist,sparsity,segnums,end, explanation.Target_class_prob, explanation.flip] # ,
        lgr.info(",".join(repr(e) for e in tmp))


        print("done on instance ", instance)
    cfs = np.save("./logs/ALIBI/" + name + '_cfs.npy', cfs)
    lgr.removeHandler(fh)
    fh.close()
    tf.keras.backend.clear_session()


