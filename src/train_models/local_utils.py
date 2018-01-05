import os, sys, gc, pickle, math, random, itertools, subprocess, multiprocessing
import datetime
from functools import reduce
from tqdm import tqdm
from joblib import Parallel, delayed

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics as metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

# Keras imports
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras.regularizers import *
from keras.constraints import *
from keras.utils.conv_utils import conv_output_length
from keras.engine import InputSpec, Layer
from keras.models import *
from keras.layers import *
from keras.layers.core import Activation
from keras.losses import *
from keras.metrics import *
from keras.callbacks import *
from keras.utils import Sequence, OrderedEnqueuer, plot_model, normalize
from keras.utils.vis_utils import model_to_dot
from keras.utils.generic_utils import Progbar

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

cpu_cores = multiprocessing.cpu_count()
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

data_dir = '../../data/'
models_dir = '../../models/'
results_dir = models_dir+'predictions/'
blend_dir = models_dir+'blend/'

# Seed
seed = 7961730
print("Seed: {}".format(seed))

def load_data():
    df = pd.read_csv(data_dir + 'train_labels.csv', index_col='filename')

    Y = df.values
    video_names = np.array(df.index)

    inx2label = [col for col in df.columns]
    label2inx = {col: i for i, col in enumerate(inx2label)}

    test_df = pd.read_csv(data_dir + 'submission_format.csv', index_col='filename')
    test_video_names = np.array(test_df.index)

    # Load meta_data
    video_meta_df = pd.read_csv(data_dir + 'video_metadata.csv', index_col='video_id')[['height', 'width']]
    file_sizes_df = pd.read_csv(data_dir + 'file_sizes.csv', index_col='video_id')[['size']]

    meta_df = video_meta_df.join(file_sizes_df)

    del video_meta_df
    del file_sizes_df

    meta_df = meta_df[meta_df.index.isin(list(video_names) + list(test_video_names))]
    meta_df['size'] = meta_df.groupby(['height', 'width']).transform(lambda x: (x - x.mean()) / x.std())
    meta_df = pd.get_dummies(meta_df, columns=['height', 'width'], drop_first=True)

    X_meta = meta_df[meta_df.index.isin(video_names)].sort_index().values
    test_X_meta = meta_df[meta_df.index.isin(test_video_names)].sort_index().values

    meta_data = pickle.load(open(data_dir + 'meta_data.pkl', 'rb'))
    test_meta_data = pickle.load(open(data_dir + 'test_meta_data.pkl', 'rb'))

    meta_mean = np.concatenate((meta_data, test_meta_data)).mean(axis=0)
    meta_std = np.concatenate((meta_data, test_meta_data)).std(axis=0)

    X_meta = np.concatenate([X_meta, (meta_data - meta_mean)/meta_std], axis=-1)
    test_X_meta = np.concatenate([test_X_meta, (test_meta_data - meta_mean)/meta_std], axis=-1)
    
    return video_names, X_meta, Y, test_video_names, test_X_meta, inx2label, label2inx

def load_npy_files(folder, names, shape, cpu_cores=cpu_cores):
    return np.stack(Parallel(n_jobs=cpu_cores)(delayed(load_npy)(folder, name, shape) for name in names))
        
def load_npy(folder, name, shape):
    try:
        arr = np.load(file = '{}{}'.format(os.path.join(folder, name),'.npy'))
        if arr.shape != shape:
            return np.zeros(shape)
        else:
            return arr
    except:
        print("Cannot read {}{}".format(folder, name))
        return np.zeros(shape)

class FeatureSequence(Sequence):
    
    def __init__(self, folders, names, shapes, X_meta, Y, batch_size, 
                 shuffle=False, parallel=False, workers=cpu_cores):
        
        self.folders = folders if (type(folders)==list) else [folders] 
        self.shapes = shapes if (type(shapes)==list) else [shapes]
        assert len(self.folders)==len(self.shapes), "number of shapes should be equal to number of folders"
        
        self.names = names
        self.X_meta, self.Y = X_meta, Y
        self.batch_size = batch_size
        
        self.inx = np.arange(0, self.names.shape[0])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)
            
        self.parallel = parallel
        self.workers = workers

    def __len__(self):
        return math.ceil(self.names.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        inputs = []
        
        if self.parallel:
            for folder, shape in zip(self.folders, self.shapes):
                features = load_npy_files(folder, self.names[batch_inx], shape, self.workers)
                inputs.append(features)
        else:
            for folder, shape in zip(self.folders, self.shapes):
                features = np.empty((batch_inx.shape[0], shape[0], shape[1]), dtype=np.float32)
                for inx, name in enumerate(self.names[batch_inx]):
                    features[inx] = load_npy(folder, name, shape)
                inputs.append(features)
        
        return [*inputs, self.X_meta[batch_inx]], self.Y[batch_inx]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
        gc.collect()
        
class PseudoFeatureSequence(Sequence):
    
    def __init__(self, folders, names, shapes, X_meta, Y, batch_size, 
                 test_folders, test_names, test_X_meta, test_Y, test_batch_size,
                 shuffle=False, parallel=False, workers=cpu_cores):
        
        self.folders = folders if (type(folders)==list) else [folders]
        self.test_folders = test_folders if (type(test_folders)==list) else [test_folders] 
        self.shapes = shapes if (type(shapes)==list) else [shapes]
        
        assert len(self.folders)==len(self.shapes), "number of shapes should be equal to number of folders"
        assert len(self.folders)==len(self.test_folders), "number of test_folders should be equal to number of folders"
        
        self.names = names
        self.X_meta, self.Y = X_meta, Y
        self.batch_size = batch_size
        self.inx = np.arange(0, self.names.shape[0])
        
        self.test_names = test_names
        self.test_X_meta, self.test_Y = test_X_meta, test_Y
        self.test_batch_size = test_batch_size
        self.test_inx = np.arange(0, self.test_names.shape[0])
        
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)
            np.random.shuffle(self.test_inx)
            
        self.parallel = parallel
        self.workers = workers

    def __len__(self):
        return math.ceil(self.names.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        test_batch_inx = self.test_inx[i*self.test_batch_size:(i+1)*self.test_batch_size]
        
        inputs = []
        if self.parallel:
            for folder, test_folder, shape in zip(self.folders, self.test_folders, self.shapes):
                features = load_npy_files(folder, self.names[batch_inx], shape, self.workers)
                test_features = load_npy_files(test_folder, self.test_names[test_batch_inx], shape, self.workers)
                
                inputs.append(np.concatenate((features, test_features)))
        else:
            for folder, test_folder, shape in zip(self.folders, self.test_folders, self.shapes):
                features = np.empty((batch_inx.shape[0], shape[0], shape[1]), dtype=np.float32)
                for inx, name in enumerate(self.names[batch_inx]):
                    features[inx] = load_npy(folder, name, shape)

                test_features = np.empty((test_batch_inx.shape[0], shape[0], shape[1]), dtype=np.float32)
                for inx, name in enumerate(self.test_names[test_batch_inx]):
                    test_features[inx] = load_npy(test_folder, name, shape)
                    
                inputs.append(np.concatenate((features, test_features)))
        
        return ([*inputs, np.concatenate((self.X_meta[batch_inx], self.test_X_meta[test_batch_inx]))], 
                np.concatenate((self.Y[batch_inx], self.test_Y[test_batch_inx])))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
            np.random.shuffle(self.test_inx)
        gc.collect()

def compute_losses(Y_true, Y_pred, eps=1e-15):
    losses = []
    for cl in range(0, Y_true.shape[1]):
        losses.append(metrics.log_loss(Y_true[:,cl], Y_pred[:,cl], labels=[0, 1], eps=eps))
        
    return losses

def find_opt_clip(y_true, y_pred, cl_inx, eps=1e-15, 
                  min_grid=np.arange(0.00001, 0.0001, 0.00001),
                  max_grid=np.arange(0.95, 0.9999, 0.0001)):
    
    opt_min_clip = 0.00001
    opt_max_clip = 0.95  
    
    max_loss = metrics.log_loss(y_true, clip(y_pred, opt_min_clip, opt_max_clip), labels=[0, 1], eps=eps)
    for min_clip in min_grid:
        loss = metrics.log_loss(y_true, clip(y_pred, min_clip, opt_max_clip), labels=[0, 1], eps=eps)
        if loss < max_loss:
            max_loss = loss
            opt_min_clip = min_clip
       
    max_loss = metrics.log_loss(y_true, clip(y_pred, opt_min_clip, opt_max_clip), labels=[0, 1], eps=eps)
    for max_clip in max_grid:
        loss = metrics.log_loss(y_true, clip(y_pred, opt_min_clip, max_clip), labels=[0, 1], eps=eps)
        if loss < max_loss:
            max_loss = loss
            opt_max_clip = max_clip
    
    print("{}: loss {} with {},{}".format(cl_inx, max_loss, opt_min_clip, opt_max_clip))
    return (cl_inx, [opt_min_clip, opt_max_clip])

def find_opt_clip_map(Y_true, Y_pred, cpu_cores=cpu_cores):
    clips_map = dict(Parallel(n_jobs=cpu_cores)(delayed(find_opt_clip)(Y_true[:,cl_inx], Y_pred[:,cl_inx], cl_inx) 
                                                for cl_inx in np.arange(0, Y_true.shape[1])))
    return [clips_map[i] for i in range(0,Y_true.shape[1])]

def clip(arr, min_value=0.001, max_value=0.95):
    return np.clip(arr, min_value, max_value)

def stratified_sampling(Y, split=0.1, random_state=seed):
    train_inx = []
    valid_inx = []
    
    n_classes = Y.shape[1]
    inx = np.arange(Y.shape[0])

    for i in range(0,n_classes):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=random_state+i)
        b_train_inx, b_valid_inx = next(sss.split(inx, Y[:,i]))
        # to ensure there is no repetetion within each split and between the splits
        train_inx = train_inx + list(set(list(b_train_inx)) - set(train_inx) - set(valid_inx))
        valid_inx = valid_inx + list(set(list(b_valid_inx)) - set(train_inx) - set(valid_inx))
        
    return np.array(train_inx), np.array(valid_inx)

def stratified_kfold_sampling(Y, n_splits=10, random_state=seed):
    train_folds = [[] for _ in range(n_splits)]
    valid_folds = [[] for _ in range(n_splits)]

    n_classes = Y.shape[1]
    inx = np.arange(Y.shape[0])
    valid_size = 1.0 / n_splits

    for cl in range(0, n_classes):
        sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state+cl)
        
        for fold, (train_index, test_index) in enumerate(sss.split(inx, Y[:,cl])):
            b_train_inx, b_valid_inx = inx[train_index], inx[test_index]
            
            # to ensure there is no repetetion within each split and between the splits
            train_folds[fold] = train_folds[fold] + list(set(list(b_train_inx)) - set(train_folds[fold]) - set(valid_folds[fold]))
            valid_folds[fold] = valid_folds[fold] + list(set(list(b_valid_inx)) - set(train_folds[fold]) - set(valid_folds[fold]))
        
    return np.array(train_folds), np.array(valid_folds)

def plot_stratified_sampling(Y, train_inx, valid_inx, labels_names, height=12, width=14):
    n_classes = Y.shape[1]
    train_count = []
    valid_count = []
    dist = []
    
    #checking distribution for each class
    for i in range(0, n_classes):
        trn_uniq = np.unique(Y[train_inx,i],return_counts=True)
        if 1.0 in trn_uniq[0]:
            train_count.append(trn_uniq[1][1])
        else:
            train_count.append(0)

        val_uniq = np.unique(Y[valid_inx,i],return_counts=True)
        if 1.0 in val_uniq[0]:
            valid_count.append(val_uniq[1][1])
        else:
            valid_count.append(0)

        dist.append(train_count[-1]/len(train_inx))
        dist.append(valid_count[-1]/len(valid_inx))

    dist_labels = [x for pair in zip([x + '_trn' for x in labels_names],
                                     [x + '_val' for x in labels_names]) for x in pair]

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.barh(np.arange(len(dist)), dist)
    ax.set_yticks(np.arange(len(dist)))
    ax.set_yticklabels(dist_labels)
    ax.invert_yaxis()
    ax.set_xlabel('%')
    ax.grid()

    for i, v in enumerate(zip(dist, list(itertools.chain(*zip(train_count, valid_count))))):
        ax.text(v[0] + 0.001, i + 0.25, str(v[1]))

    plt.show()

def plot_stratified_kfold_sampling(Y, train_folds, valid_folds, labels_names, height=12, width=14):
    for train_inx, valid_inx in zip(train_folds, valid_folds):
        plot_stratified_sampling(Y, train_inx, valid_inx, labels_names, height, width)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 2.0.6
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None