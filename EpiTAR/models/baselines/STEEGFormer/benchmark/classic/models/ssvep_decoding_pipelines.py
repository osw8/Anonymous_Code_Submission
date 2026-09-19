"""
This file contains two pipelines tailored for ssvep classification. 
Note that FBCCA is not ideal for binocular-swap vision as half of the targets share the same stimulation frequencies. 
We still include this model in our analysis to study the impact of train-free methods.

Reference:
    [1] Demo code provided in https://gigadb.org/dataset/102557
    [2] https://github.com/nbara/python-meegkit/blob/master/meegkit/trca.py
"""


import numpy as np
import time

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import CCA

from scipy.special import softmax
from numpy import linalg as LA
from meegkit.trca import trca
from mne.filter import filter_data

class Scaler3D(BaseEstimator, TransformerMixin):
    """
    standardize the dataset
    """
    def fit(self, X, y=None):
        self.scalers_ = {}
        n_channels, n_times = X.shape[1], X.shape[2]
        for ch in range(n_channels):
            for t in range(n_times):
                values = X[:, ch, t]
                self.scalers_[(ch, t)] = (values.mean(), values.std())
        return self

    def transform(self, X):
        X_scaled = np.empty_like(X)
        n_channels, n_times = X.shape[1], X.shape[2]
        for ch in range(n_channels):
            for t in range(n_times):
                mean, std = self.scalers_[(ch, t)]
                X_scaled[:, ch, t] = (X[:, ch, t] - mean) / std
        return X_scaled


class TRCAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, fs=250, ensemble=True):
        self.template = {}
        self.fs = fs
        self.ensemble = ensemble
        
    def fit(self, X, y):
        _, n_chans, self.n_bands, n_times = X.shape
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        
        self.filters_ = np.zeros((self.n_bands, self.n_classes, n_chans))
        self.template = np.zeros((self.n_classes, self.n_bands, n_times, n_chans))
        
        
        for class_i in range(self.n_classes):
            X_cls = X[y == self.classes_[class_i]]  
            X_cls = X_cls.transpose(2, 3, 1, 0) 
            for fb_i in range(self.n_bands):
                X_cls_band = X_cls[fb_i] 
                w_best = trca(X_cls_band)
                
                self.filters_[fb_i, class_i, :] = w_best
                self.template[class_i, fb_i] = np.mean(X_cls_band, axis=2)  
            
        return self
        
    def predict(self, X):
        
        X = X.transpose(2, 3, 1, 0) 
        _, _, _, n_trials = X.shape
        
        
        fb_coefs = [(b + 1)**(-1.25) + 0.25 for b in range(self.n_bands)]
        
        r = np.zeros((self.n_bands, self.n_classes))
        preds = np.zeros((n_trials), "int")  

        for trial in range(n_trials):
            test_tmp = X[..., trial]  
            for fb_i in range(self.n_bands):
                
                testdata = test_tmp[fb_i]

                for class_i in range(self.n_classes):
                    
                    
                    template = np.squeeze(self.template[class_i, fb_i])
                    
                    if self.ensemble:
                        
                        w = np.squeeze(self.filters_[fb_i]).T
                    else:
                        
                        w = np.squeeze(self.filters_[fb_i, class_i])

                    
                    
                    
                    r_tmp = np.corrcoef((testdata @ w).flatten(),
                                        (template @ w).flatten())
                    r[fb_i, class_i] = r_tmp[0, 1]

            rho = np.dot(fb_coefs, r)  
            
            tau = np.argmax(rho)  
            preds[trial] = int(tau)

        return preds

    def predict_proba(self, X):
        X = X.transpose(2, 3, 1, 0) 
        _, _, _, n_trials = X.shape
        
        
        fb_coefs = [(b + 1)**(-1.25) + 0.25 for b in range(self.n_bands)]
        
        r = np.zeros((self.n_bands, self.n_classes))
        probs = np.zeros((n_trials, self.n_classes))

        for trial in range(n_trials):
            test_tmp = X[..., trial]  
            for fb_i in range(self.n_bands):
                
                testdata = test_tmp[fb_i]
                
                for class_i in range(self.n_classes):
                    
                    
                    template = np.squeeze(self.template[class_i, fb_i])
                    
                    if self.ensemble:
                        
                        w = np.squeeze(self.filters_[fb_i]).T
                    else:
                        
                        w = np.squeeze(self.filters_[fb_i, class_i])

                    
                    
                    r_tmp = np.corrcoef((testdata @ w).flatten(),
                                        (template @ w).flatten())
                    r[fb_i, class_i] = r_tmp[0, 1]
                    
                
            rho = np.dot(fb_coefs, r)  
            
            prob = softmax(rho)  
            probs[trial] = prob

        return probs


def cal_CCA_regularized(X, Y, reg=1e-6):
    """
    Canonical Correlation Analysis with regularization.
    :param X: (n_samples x n_channels_X)
    :param Y: (n_samples x n_channels_Y)
    :param reg: regularization strength (default 1e-6)
    :return: U, V, corr_cca
    """

    
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    
    YtY = Y.T @ Y + reg * np.eye(Y.shape[1])
    XtX = X.T @ X + reg * np.eye(X.shape[1])

    P_Y = Y @ np.linalg.inv(YtY) @ Y.T
    b_U = np.linalg.inv(XtX) @ (X.T @ P_Y @ X)
    eigvals_U, eigvecs_U = np.linalg.eig(b_U)
    U = eigvecs_U[:, np.argmax(eigvals_U.real)]

    
    P_X = X @ np.linalg.inv(XtX) @ X.T
    b_V = np.linalg.inv(YtY) @ (Y.T @ P_X @ Y)
    eigvals_V, eigvecs_V = np.linalg.eig(b_V)
    V = eigvecs_V[:, np.argmax(eigvals_V.real)]

    
    x_proj = X @ U
    y_proj = Y @ V
    corr = np.corrcoef(x_proj.T, y_proj.T)
    corr_cca = corr[0, 1]

    return U.real, V.real, corr_cca.real
    
class FBDCCAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, fs=250, n_harmonics=5):
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.weights = np.array([1, 0.71, 0.58, 0.5, 0.45, 0.41, 0.38, 0.35, 0.33, 0.31])
        
        
        self.f1 = [14.74, 12.42, 12.47, 15.33, 12.88, 10.1, 8.51, 11.67, 15.4, 12.36, 8.81, 11.33, 11.01,
            9.52, 11.22, 11.92, 11.62, 14.72, 8.62, 14.43, 15.59, 10.9, 8.57, 13.93, 15.04, 15.97, 14.37, 8.21,
            11.97, 12.94, 9.57, 12.45, 10.94, 13.11, 9.49, 12.63, 12.41, 15.91, 13.34, 10.62]
        self.f2 = [10.1, 12.63, 12.88, 11.22, 12.47, 14.74, 8.57, 11.62, 13.11, 12.45, 8.21, 11.97, 10.94, 12.41,
            15.33, 13.34, 11.67, 15.59, 10.62, 12.94, 14.72, 13.93, 8.51, 10.9, 14.37, 15.91, 15.04, 8.81, 11.33,
            14.43, 9.49, 12.36, 11.01, 15.4, 9.57, 12.42, 9.52, 15.97, 11.92, 8.62]
        
    def fit(self, X, y):
        _, _, _, n_times = X.shape
        self.n_classes = len(np.unique(y))
        
        
        template = []
        time = np.arange(n_times) / self.fs
        
        for f in range(self.n_classes):
            template_target = []
            for harmonic in range(1, self.n_harmonics + 1):
                template_target.append(np.sin(2 * np.pi * harmonic * self.f1[f] * time))
                template_target.append(np.cos(2 * np.pi * harmonic * self.f1[f] * time))
                template_target.append(np.sin(2 * np.pi * harmonic * self.f2[f] * time))
                template_target.append(np.cos(2 * np.pi * harmonic * self.f2[f] * time))
                    
            template_target = np.array(template_target)
            template.append(template_target)
        self.template = template
        return self
        
    def predict(self, X):
        n_trials, _, _, _ = X.shape
        preds = np.zeros((n_trials), "int")  
        for trial in range(n_trials):
            test_tmp = X[trial,:,:, :]  
            rr = {}
            
            for fb_i in range(self.n_harmonics):
                rr[str(fb_i)] = []
                for target_num in range(self.n_classes):
                    
                    testdata = np.squeeze(test_tmp[:, fb_i, :])
                    _, _, corr = cal_CCA_regularized(testdata.T, self.template[target_num].T, reg=1e-5)
                    rr[str(fb_i)].append(corr)
                    
            
            rr['all'] = np.zeros(self.n_classes)
            for fb_i in range(self.n_harmonics):
                rr['all'] += np.square(np.array(rr[str(fb_i)])) * self.weights[fb_i]
                
            tau = np.argmax(rr['all']) 
            preds[trial] = int(tau)
            
        return preds
                            
                        
    def predict_proba(self, X):
        n_trials, _, _, _ = X.shape
        
        probs = np.zeros((n_trials, self.n_classes))
        for trial in range(n_trials):
            
            test_tmp = X[trial,:,:, :]  
            rr = {}
            
            for fb_i in range(self.n_harmonics):
                
                start = time.time()
                rr[str(fb_i)] = []
                for target_num in range(self.n_classes):
                    
                    
                    testdata = np.squeeze(test_tmp[:, fb_i, :])
                    _, _, corr = cal_CCA_regularized(testdata.T, self.template[target_num].T, reg=1e-5)
                    rr[str(fb_i)].append(corr)
                
                end = time.time()
                print(f'test time: {end - start}')
            
            
            rr['all'] = np.zeros(self.n_classes)
            for fb_i in range(self.n_harmonics):
                rr['all'] += np.square(np.array(rr[str(fb_i)])) * self.weights[fb_i]
            
            prob = softmax(rr['all'])  
            probs[trial] = prob
            
        return probs
        
        

def count_pipeline_params(pipe):
    
    
    
    
    total = 0
    attrs = ['coef_', 'intercept_', 'means_', 'covmeans_', 'filters_']
    
    for name, step in pipe.named_steps.items():
        modules = [step]
        for attr in dir(step):
            if attr.startswith('_') and not attr.startswith('__'):
                submod = getattr(step, attr)
                if hasattr(submod, '__class__'):
                    modules.append(submod)

        for mod in modules:
            for attr in attrs:
                if hasattr(mod, attr):
                    val = getattr(mod, attr)
                    try:
                        if hasattr(val, 'size'):
                            total += val.size
                    except Exception:
                        continue

    return total


def pipe_trca(fs = 250):
    pipe = make_pipeline(TRCAClassifier(fs = fs))
    return pipe

def pipe_cca(fs = 250):
    pipe = make_pipeline(FBDCCAClassifier(fs = fs))
    return pipe
