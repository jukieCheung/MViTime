import warnings
import numpy as np
from scipy.signal import resample
# import pywt
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
# ===========================================
warnings.filterwarnings("ignore")
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.cpu().detach().numpy()
    y_pre = np.argmax(y_pre, axis=-1)
    return f1_score(y_true, y_pre, average='macro')

def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
# =======================================
def sig_wt_filt(sig):

    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt


def multi_prep(sig, target_point_num=1280):

    assert len(sig.shape) == 2, 'Not for 1-D data.Use 2-D data.'
    sig = resample(sig, target_point_num, axis=1)
    for i in range(sig.shape[0]):
        sig[i] = sig_wt_filt(sig[i])
    sig = scale(sig, axis=1)
    return sig


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    y_true_argmax = np.argmax(y_true,axis=1)
    y_pred = np.array(y_pred)
    #y_pred_argmax = np.argmax(y_pred,axis=1)
    cm = confusion_matrix(y_true_argmax, y_pred)
    #print(y_pred)

    #classes = classes[unique_labels(y_true_argmax.tolist(), y_pred.tolist())]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    return cm


def print_results(y_true, y_pred, target_names):
 
    overall_accuracy = accuracy_score(y_true, y_pred)
    print('\n----- overall_accuracy: {0:f} -----'.format(overall_accuracy))
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(target_names)):
        print(target_names[i] + ':')
        Se = cm[i][i]/np.sum(cm[i])
        Pp = cm[i][i]/np.sum(cm[:, i])
        print('  Se = ' + str(Se))
        print('  P+ = ' + str(Pp))
    print('--------------------------------------')
