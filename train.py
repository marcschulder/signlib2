import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from keras.layers import SimpleRNN
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.layers.normalization import BatchNormalization
# from keras.layers import GRU
from keras.layers.wrappers import Bidirectional
# from sklearn.utils import class_weight
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import misc2


# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.get_cmap('Paired'))
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def split_train_test(df, window):
    # print(df)

    for n in range(df["Windowed_poses"].shape[0]):
        df["Windowed_poses"].iloc[n] = np.hstack(df["Windowed_poses"].iloc[n]).reshape((window, 14))

    data = np.hstack(df["Windowed_poses"]).reshape((df["Windowed_poses"].shape[0], window, 14))

    labels = df["Class"].values.flatten()
    # Change if binary
    # labels = labels.reshape((df["Class"].shape[0],1))
    labels = to_categorical(labels, num_classes=3)

    #     my_labels = []
    #     for ii in range(labels.shape[0]):
    #         my_labels.append(labels[ii][0])

    print("shape of data: ", data.shape)
    print("shape of values: ", labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42,
                                                        stratify=labels)
    return (X_train, X_test, y_train, y_test)


def lstm_train(window, data_h5):
    # lstm_dim = [8,16,32,64,128,256]
    lstm_dim = [16, 32]
    mf = pd.DataFrame(columns=['Window', 'lstm_dim', 'Test_score', 'Test_accuracy'])
    df = misc2.create_dataframe_with_window(data_h5, window)

    # print(df)

    for n in range(df["Windowed_poses"].shape[0]):
        df["Windowed_poses"].iloc[n] = np.hstack(df["Windowed_poses"].iloc[n]).reshape((window, 14))

    data = np.hstack(df["Windowed_poses"]).reshape((df["Windowed_poses"].shape[0], window, 14))

    labels = df["Class"].values.flatten()
    # Change if binary
    # labels = labels.reshape((df["Class"].shape[0],1))
    labels = to_categorical(labels, num_classes=3)

    #     my_labels = []
    #     for ii in range(labels.shape[0]):
    #         my_labels.append(labels[ii][0])

    print("shape of data: ", data.shape)
    print("shape of values: ", labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42,
                                                        stratify=labels)

    for l in lstm_dim:
        model = Sequential()

        # model.add(BatchNormalization(input_shape=(window,14)))

        # lstm 128 works 88%
        model.add(Bidirectional(LSTM(l, input_shape=(window, 14), activation='tanh', return_sequences=False)))
        # model.add(SimpleRNN(32, activation='relu', return_sequences=False))

        # model.add(Dense(5,activation='tanh'))
        model.add(Dense(labels.shape[1], activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # early stop
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=0)
        # 75 epochs seems the best one. with lstm 128
        history = model.fit(X_train, y_train, batch_size=32, epochs=128, verbose=0, shuffle=False, validation_split=0.2,
                            callbacks=[es])

        score, acc = model.evaluate(X_test, y_test, batch_size=1)
        model.save('./keras_models/milti_class_classification_window_%s_lstm_%s_acc_%.2f.h5' % (window, l, acc))

        # y_pred = model.predict_classes(X_test, batch_size=1)
        #         confusion_mat = confusion_matrix(y_test, y_pred)
        #         plot_confusion_matrix(confusion_mat)
        #         target_names = ['Class-0','Class-1','Class-2']
        #         print(classification_report(y_test, y_pred, target_names=target_names))
        mf = mf.append(
            pd.DataFrame({"Window": window, "lstm_dim": l, "Test_score": score, "Test_accuracy": acc * 100}, index=[0]))
    return (mf, X_test, y_test)


def lstm_train_binary(window, data_h5):
    lstm_dim = [64, 128, 256]
    from sklearn.metrics import roc_auc_score

    mf = pd.DataFrame(columns=['Window', 'lstm_dim', 'Test_score', 'Test_accuracy'])

    df = misc2.create_dataframe_with_window(data_h5, window)

    ## other approach
    #     df["Windowed_poses"] = pd.Series(df["Windowed_poses"]).apply(np.asarray)
    #     max_sequence_length = 14
    #     X_init = np.asarray(df.Windowed_poses)
    #     # Use hstack to and reshape to make the inputs a 3d vector
    #     X_t = np.hstack(X_init).reshape(df["Windowed_poses"].iloc[n],window,14)
    #     #y_t = np.hstack(np.asarray(original_data.Handeness)).reshape(264,)

    # data = X_t

    ##

    for n in range(df["Windowed_poses"].shape[0]):
        df["Windowed_poses"].iloc[n] = np.hstack(df["Windowed_poses"].iloc[n]).reshape((window, 14))

    data = np.hstack(df["Windowed_poses"]).reshape((df["Windowed_poses"].shape[0], window, 14))

    labels = df["Class"].values.flatten()
    # Change if binary
    # labels = labels.reshape((df["Class"].shape[0],1))
    labels = to_categorical(labels)

    print("shape of data: ", data.shape)
    print("shape of values: ", labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    #     class_weights = class_weight.compute_class_weight('balanced',
    #                                                  np.unique(y_train),
    #                                                  y_train[:,0])

    for l in lstm_dim:
        model = Sequential()
        model.add(BatchNormalization(input_shape=(window, 14)))

        # lstm 128 works 88%
        model.add(Bidirectional(LSTM(l, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=False)))
        model.add(Dense(7, activation='sigmoid'))
        model.add(Dense(labels.shape[1], activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 75 epochs seems the best one. with lstm 128
        history = model.fit(X_train, y_train, batch_size=64, epochs=64, verbose=0, shuffle=False, validation_split=0.1)

        score, acc = model.evaluate(X_test, y_test, batch_size=16)
        probs = model.predict(X_test, batch_size=1)
        # print(y_test.dtype)
        roc = roc_auc_score(y_test, probs)
        mf = mf.append(
            pd.DataFrame({"Window": window, "lstm_dim": l, "Test_score": acc, "Test_accuracy": roc}, index=[0]))
        model.save('./keras_models/binary_classification_window_%s_lstm_%s_roc_%.2f.h5' % (window, l, roc))
    return (mf)


def train_rnn_tensorflow(window, data_h5):
    import tensorflow.compat.v1 as tf

    # mf = pd.DataFrame(columns=['Window', 'lstm_dim', 'Test_score', 'Test_accuracy'])

    df = misc2.create_dataframe_with_window(data_h5, window)

    for n in range(df["Windowed_poses"].shape[0]):
        df["Windowed_poses"].iloc[n] = np.hstack(df["Windowed_poses"].iloc[n]).reshape((window, 14))

    data = np.hstack(df["Windowed_poses"]).reshape((df["Windowed_poses"].shape[0], window, 14))

    labels = df["Class"].values.flatten()
    # Change if binary
    # labels = labels.reshape((df["Class"].shape[0],1))
    # labels = to_categorical(labels)

    print("shape of data: ", data.shape)
    print("shape of values: ", labels.shape)
    print(data[0])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=False)

    n_steps = window
    n_inputs = 14
    n_neurons = 150
    n_outputs = 3

    learning_rate = 0.001
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    n_epochs = 100
    batch_size = 150
    cc = 0
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(X_train.shape[0] // batch_size):
                X_batch = X_train[cc:(cc + batch_size)]
                y_batch = y_train[cc:(cc + batch_size)]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                cc = cc + 150
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy: ", acc_train, "Test_accuracy: ", acc_test)
