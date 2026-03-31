import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import load_model
import numpy as np
import time
from alibi.explainers import tfcounterfactual_timex
from tslearn.barycenters import dtw_barycenter_averaging
import sklearn
from sklearn import preprocessing
import warnings
from scipy.spatial import distance
import logging
import fileHandlerWithHeader
import os
warnings.simplefilter("ignore", UserWarning)

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
for DS in ['Coffee']:
    lgr = logging.getLogger('LogDS')
    lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
    # add a file handler #     # ACF_distProto
    fh = fileHandlerWithHeader.FileHandlerWithHeader("./logs/TimeX/"+DS + '_LOG_0.csv',
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

    print("Number of Classes "+str(nb_classes))
    dba_dic = {}
    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((ytrain, ytest_), axis=0).reshape(-1, 1))
    ytrain = enc.transform(ytrain.reshape(-1, 1)).toarray()
    ytest = enc.transform(ytest_.reshape(-1, 1)).toarray()

    # classes = np.unique(ytest, axis=0)
    for cls in range(len(classes)):
        idx = [i for i,x in enumerate(ytrain) if x[cls:cls+1] == 1]
        tmp = np.take(xtrain, idx, 0)
        dba_dic[cls]=dtw_barycenter_averaging(tmp, max_iter=10).reshape(1,tmp.shape[1])[0]
        # plt.plot(dba_dic[cls], linestyle="--",color='red', linewidth=2)
        # plt.title("Class DBA is #: "+str(cls))
        # plt.show()

    # Transform input array into tensor
    xtrain = np.reshape(np.array(xtrain),(xtrain.shape[0], xtrain.shape[1],1))
    xtest = np.reshape(np.array(xtest),(xtest.shape[0], xtest.shape[1],1))

    path = './fcn_weights/'
    cnn = load_model(path + DS + '_best_model.hdf5')  # './fcn_weights/'
    print(cnn.summary())
    score = cnn.evaluate(xtest, ytest, verbose=0)
    print('Test accuracy: ', score[1])

    cfs = []
    for instance in range(xtest.shape[0]):
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

        otherclasses_lst = list(dba_dic.keys())
        otherclasses_lst.remove(list(ytest[instance:instance+1][0]).index(1))
        print("Test sample class is "+str(list(ytest[instance:instance+1][0]).index(1)))

        protoDistance_dic ={}
        for c in otherclasses_lst:
            tmp = sum(abs(val1-val2) for val1, val2 in zip(xtest[instance:instance + 1].reshape(1,len(dba_dic[c]))[0], dba_dic[c]))
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
            print("Explanation using prototype class "+str(c))

            cf = tfcounterfactual_timex.TFCounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,  #outlier_model=outlier_model,
                                                   target_class=c, target_classid = c,max_iter=max_iter, lam_init=lam_init,
                                                   max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                                   feature_range=feature_range, tstID=i, dataset=DS, num_classes=nb_classes, choice=i) #target_class

            explanation = cf.explain(X, dba_dic[c].reshape((1,) + xtest[0].shape))
            end = time.time() - start_time
            print("Execution Time is " + str(end))

            a = explanation.cf
            b = np.reshape(X, (1, X.shape[1]))[0]
            dist, sparsity, segnums = getmetrics(a, b)
            cfs.append(list(a))

            tmp = [dist, sparsity, segnums, end, explanation.Target_class_prob, explanation.flip]
            lgr.info(",".join(repr(e) for e in tmp))

    cfs = np.save("./logs/TimeX/" + DS + '_cfs0.npy', cfs)
    lgr.removeHandler(fh)
    fh.close()
    tf.keras.backend.clear_session()















# prototype = protoptypes_lst[:1][0][0].reshape((1,) + xtest[0].shape) #xtest[25].reshape((1,) + xtest[0].shape)
# explanation = cf.explain(X, prototype)
# print('Explanation took {:.3f} sec'.format(time() - start_time))
#
# # print(f'Counterfactual prediction: {pred_class} with probability {proba}')
# print("label of X is "+str(np.argmax(cnn.predict(X))))
# # print("label of XCF is "+str(np.argmax(cnn.predict(explanation.cf['X']))))

# # xtrain_class1 = xtrain[xtrain['y']==0][list(xtrain)[:xtrain.shape[1]-1]]
# # xtrain_class2 = xtrain[xtrain['y']==1][list(xtrain)[:xtrain.shape[1]-1]]
# # xtrain.drop(columns=['y'],inplace=True)
# # outlier_model = IsolationForest(n_estimators=1000, random_state=11).fit(xtrain_class2)
# # outlier_model.fit(xtrain_class2)


# One hot encoding of the outputs
# ytrain = to_categorical(ytrain)
# ytest = to_categorical(ytest)
#
# cnn = load_model('coffeeTS_cnn.h5')
# path = '/home/dmlab/Soukaina/Data_TS/'
# path ="/Users/soukainafilaliboubrahimi/PycharmProjects/T2G/Data_TS/"
# trdata = pd.read_csv(path+DS+"/"+DS+"_TRAIN", header=None)
# tsdata = pd.read_csv(path+DS+"/"+DS+"_TEST", header=None)
#
# # trdata = pd.read_csv("/Users/soukainafilaliboubrahimi/Downloads/timeseries/Coffee/Coffee_TRAIN", header=None)
# # tsdata = pd.read_csv("/Users/soukainafilaliboubrahimi/Downloads/timeseries/Coffee/Coffee_TEST", header=None)
#
# ytrain, xtrain = trdata[[0]], trdata[list(trdata)[1:]]
# ytest, xtest = tsdata[[0]], tsdata[list(tsdata)[1:]]
#

# def ae_model(input_shape):
#     # encoder
#     x_in = Input(input_shape)
#     x = Conv1D(filters=128, kernel_size=16, padding='same')(x_in)
#     # x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling1D(pool_size=2, padding='same')(x)
#     encoded = Conv1D(filters=256, kernel_size=32, activation=None, padding='same')(x)
#     encoder = Model(x_in, encoded)
#
#     print(encoder.summary())
#     # # decoder
#     dec_in = Input(shape=(14, 14, 1))
#     # x = Conv2D(16, (3, 3), activation='relu', padding='same')(dec_in)
#     # x = UpSampling2D((2, 2))(x)
#     # x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#     # decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
#     # decoder = Model(dec_in, decoded)
#     #
#     # # autoencoder = encoder + decoder
#     # x_out = decoder(encoder(x_in))
#     # autoencoder = Model(x_in, x_out)
#     # autoencoder.compile(optimizer='adam', loss='mse')
#
#     return encoder
#     # return autoencoder, encoder, decoder
#
#
# # ae_model(xtrain.shape[1:])
#
# def build_model(input_shape, nb_classes):
#     input_layer = Input(input_shape)
#     conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Activation(activation='relu')(conv1)
#
#     conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Activation('relu')(conv2)
#
#     # conv3 = Conv1D(128, kernel_size=3,padding='same')(conv2)
#     # conv3 = BatchNormalization()(conv3)
#     # conv3 = Activation('relu')(conv3)
#
#     gap_layer = GlobalAveragePooling1D()(conv2)
#
#     output_layer = Dense(nb_classes, activation='softmax')(gap_layer)
#     model = Model(inputs=input_layer, outputs=output_layer)
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam',
#                   metrics=['accuracy'])
#
#     return model
#
# # cnn = build_model(xtrain.shape[1:], nclasses)
# # cnn.summary()
# # cnn.fit(xtrain, ytrain, batch_size=64, epochs=1000, verbose=1)
# # acc = cnn.predict(xtest)
# # acc = cnn.evaluate(xtest,ytest, verbose=1)
# # print("Loss is "+str(acc[0])+" , Accuracy is "+str(acc[1]))
# # cnn.save('coffeeTS_cnn.h5')
#
# # cnn = load_model('coffeeTS_cnn.h5')
# nohup python3 mainTimeX.py > timex.log 2>&1 &
# nohup python3 mainTimeX.py > timex1.log 2>&1 &
# nohup python3 mainTimeX.py > timex2.log 2>&1 &
