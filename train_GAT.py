import time
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse
import h5py
import random

from models import GAT
from models import SpGAT
from utils import process

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass

def load_hdf_data(path, network_name='network', feature_name='features'):
    """Load a GAT input HDF5 container and return its content.
    """
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EMPGI and save to file')
    parser.add_argument('-e', '--epochs', help='Number of Epochs',
                        dest='epochs',
                        default=5000,
                        type=int
                        )
    parser.add_argument('-lr', '--learningrate', help='Learning Rate',
                        dest='lr',
                        default=.001,
                        type=float
                        )
    parser.add_argument('-hd', '--hidden_dims',
                        help='Hidden Dimensions (number of filters per layer). Also determines the number of hidden layers.',
                        nargs='+',
                        dest='hidden_dims',
                        default=[8, 4])
    parser.add_argument('-ah', '--atten_heads',
                        help='Num of attention head in each layer, additional entry for the output layer',
                        nargs='+',
                        dest='atten_heads',
                        default=[8, 4, 1])
    parser.add_argument('-lm', '--loss_mul',
                        help='Number of times, false negatives are weighted higher than false positives',
                        dest='loss_mul',
                        default=45,
                        type=float
                        )
    parser.add_argument('-wd', '--weight_decay', help='Weight Decay',
                        dest='decay',
                        default=5e-4,
                        type=float
                        )
    parser.add_argument('-seed', '--seed', help='Random seed',
                        dest='seed',
                        default=21,
                        type=int
                        )
    parser.add_argument('-do', '--dropout', help='Dropout Percentage',
                        dest='dropout',
                        default=.5,
                        type=float
                        )
    parser.add_argument('-d', '--data', help='Path to HDF5 container with data',
                        dest='data',
                        type=str,
                        required=True
                        )
    args = parser.parse_args()
    return args

args = parse_args()

set_seed(args.seed)

dataset = args.data

# training params
batch_size = 1
nb_epochs = args.epochs
lr = args.lr  # learning rate
l2_coef = args.decay  # weight decay
loss_mul = args.loss_mul
dropout = args.dropout
hid_units = [int(x) for x in args.hidden_dims] # numbers of hidden units per each attention head in each layer
n_heads = [int(x) for x in args.atten_heads] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = SpGAT

# checkpt_file = 'saved_model/mod_{}.ckpt'.format(dataset)

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('seed: '+str(args.seed))
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('loss_mul: ' + str(loss_mul))
print('Dropout: ' + str(dropout))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

sparse = True

data = load_hdf_data(dataset, feature_name='features')
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data

# features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]


features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

adj = sp.coo_matrix(adj) # convert to sparse matrix
biases = process.preprocess_adj_bias(adj)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        if sparse:
            #bias_idx = tf.placeholder(tf.int64)
            #bias_val = tf.placeholder(tf.float32)
            #bias_shape = tf.placeholder(tf.int64)
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
        else:
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_cross_entropy_weight(log_resh, lab_resh, msk_resh, loss_mul)

    score = tf.nn.sigmoid(log_resh)
    
    accuracy = model.masked_accuracy(score, lab_resh, msk_resh)

    performance_ops = model.get_performance_metrics(score, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    running_avg_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                            scope="evaluation")
    metric_reset_op = tf.variables_initializer(var_list=running_avg_vars)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        train_aupr_avg = 0
        train_auroc_avg = 0
        
        val_acc_avg = 0
        val_aupr_avg = 0
        val_auroc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]
            sess.run( tf.local_variables_initializer() )

            while tr_step * batch_size < tr_size:
                feed_dict = {ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                    bias_in: biases,
                    lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                    msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                    is_train: True,
                    attn_drop: dropout, ffd_drop: dropout}

                
                _, loss_value_tr = sess.run([train_op, loss], feed_dict=feed_dict)

                acc_tr, aupr_tr, auroc_tr = sess.run(performance_ops, feed_dict=feed_dict)

                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                train_aupr_avg += aupr_tr
                train_auroc_avg += auroc_tr
                tr_step += 1  
            #np.save('y_pred', y_pred)

            vl_step = 0
            vl_size = features.shape[0]
            #sess.run( tf.local_variables_initializer() )
            sess.run(metric_reset_op)

            while vl_step * batch_size < vl_size:
                acc_vl, aupr_vl, auroc_vl = sess.run(performance_ops,
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases,
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})

                val_acc_avg += acc_vl
                val_aupr_avg += aupr_vl
                val_auroc_avg += auroc_vl
                vl_step += 1

            print('Epoch %d Training: loss = %.4f, acc = %.4f aupr = %.4f auroc = %.4f | Val: acc = %.4f aupr = %.4f auroc = %.4f' %
                 (epoch + 1, train_loss_avg/tr_step, train_acc_avg/tr_step, train_aupr_avg/tr_step, train_auroc_avg/tr_step,
                 val_acc_avg/vl_step, val_aupr_avg/vl_step, val_auroc_avg/vl_step))

            # if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
            #     if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
            #         vacc_early_model = val_acc_avg/vl_step
            #         vlss_early_model = val_loss_avg/vl_step
            #         saver.save(sess, checkpt_file)
            #     vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
            #     vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
            #     curr_step = 0
            # else:
            #     curr_step += 1
            #     if curr_step == patience:
            #         print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
            #         print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
            #         break

            train_loss_avg = 0
            train_acc_avg = 0
            train_aupr_avg = 0
            train_auroc_avg = 0

            #val_loss_avg = 0
            val_acc_avg = 0
            val_aupr_avg = 0
            val_auroc_avg = 0


        sess.run( tf.local_variables_initializer() )

        ts_size = features.shape[0]
        ts_step = 0

        while ts_step * batch_size < ts_size:
            y_test_pred = sess.run(score,
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases,
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            acc_ts, aupr_ts, auroc_ts = sess.run(performance_ops,
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases,
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})

            ts_step += 1

        print('Test set acc = %.4f aupr = %.4f auroc = %.4f'%(acc_ts, aupr_ts, auroc_ts))
        np.save('y_test_pred', y_test_pred)

        sess.close()
