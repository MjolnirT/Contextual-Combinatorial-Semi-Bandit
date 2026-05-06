import numpy as np
import scipy.linalg as la

from sklearn import tree
from sklearn import ensemble

class IncrementalModel():

    def __init__(self):
        pass
        
    def fit(self, X, y):
        '''
        Fit the model to a dataset (X is n x d dimensional, where n is number of datapoints and d is data dimension).
        '''
        pass
    def predict(self, X):
        '''
        Return model predictions for each row of the matrix X
        '''
        pass
        
    def fit_incremental(self, x, keep=False):
        '''
        Update the model given a single context x.
        
        If keep is False, just return MSE. Otherwise return a copy of the new model .
        '''
        
        # TODO: Should give MSE both on the original dataset and the new dataset
        
        pass
        
    def get_dataset_size(self):
        '''
        Return size of the dataset the model was fit on (if it was fit)
        '''
        pass
        

        
class IncrementalRegressionModel(IncrementalModel):

    def get_mse(self):
        '''
        If the model has been fit, return MSE. Otherwise None
        '''
        
        pass
    
    def get_range(self):
        '''
        Return max and min values this regressor class can produce.
        '''    
            
class IncrementalLinearRegression(IncrementalRegressionModel):

    def __init__(self, reg = 0):
        
        self.reg = reg
        
        self.is_fit = False
        
    def get_dataset_size(self):
        
        assert self.is_fit
        
        return self.n
        
    def fit(self, X, y, weights = None):
        
        assert len(X.shape) == 2
        
        self.n = X.shape[0]
        self.d = X.shape[1]
       
        assert self.n == len(y)
       
        if weights is not None:
            assert self.n == len(weights)
            self.weights = weights
        else:
            self.weights = np.ones(self.n)
        self.X = X
        self.y = y
        self.y_norm = np.linalg.norm(self.y)
        
        W = np.diag(self.weights)
        
        self.A = np.eye(self.d)*self.reg + self.X.transpose().dot(W.dot(self.X))
        self.Ainv = la.pinv(self.A)
        
        self.b = self.X.transpose().dot(W.dot(self.y))
        
        # (parameter vector)
        self.theta = self.Ainv.dot(self.b)
        
        self.mse = self.weights.dot((self.X.dot(self.theta)- self.y)**2)
        
        self.is_fit = True
        
        return
        
    def get_mse(self):
    
        assert self.is_fit
        
        return self.mse
        
    def fit_incremental(self, x, y , weight = 1., keep = False):
        '''
        TODO: Fix keep option
        '''
        
        assert self.is_fit
        
        if keep is False:
            A = self.A.copy()
            Ainv = self.Ainv.copy()
            b = self.b.copy()
        else:
            A = self.A
            Ainv = self.Ainv
            b = self.b
        
        # update using sherman morrison (only works if A is invertible -- can be incorrect if regularization is turned off).
        z = Ainv.dot(x)
        
        Ainv = Ainv - weight*np.outer(z, z)/(1 + weight*x.dot(z))

        b = b + weight*x*y
        
        theta = Ainv.dot(b)
        
        if keep is True:
            self.theta = theta
        # past_pred = self.X.dot(theta)

        pred = theta.dot(x)

        # past_mse = self.weights.dot((past_pred - self.y)**2)
        past_mse = self.get_mse() + theta.dot(A.dot(theta))
        
        full_mse = past_mse + weight*(x.dot(theta) - y)**2
        
        if keep is True:
            A = A + weight*np.outer(x, x)
            self.mse = full_mse

        return (pred, past_mse, full_mse)

    def fit_incremental2(self, X, y, weights):
        '''
        TODO: Fix keep option
        '''
        
        assert self.is_fit
        
        assert len(X.shape)==2

        for idx in range(X.shape[0]):
            x = X[idx]
            y = y[idx]
            weight = weights[idx]

            # update using sherman morrison (only works if A is invertible -- can be incorrect if regularization is turned off).
            z = self.Ainv.dot(x)
        
            Ainv = self.Ainv - weight*np.outer(z, z)/(1 + weight*x.dot(z))

            b = self.b + weight*x*y
        
            theta = Ainv.dot(b)
        
            past_mse = self.get_mse() + theta.dot(self.A.dot(theta))
        
            full_mse = past_mse + weight*(x.dot(theta) - y)**2

            self.mse = full_mse
            self.A = self.A + np.outer(x, x)*weight
            self.Ainv = Ainv
            self.b = b
            self.theta = theta

        return

    def fit_incremental_slow(self, x, y , weight = 1., keep = False):
        '''
        Test version of fit_incremental that refits the whole dataset
        TODO: Fix keep option
        '''
        
        assert self.is_fit
        

        
        # update using sherman morrison (only works if A is invertible -- can be incorrect if regularization is turned off).

        Xp = np.vstack((self.X, x))
        yp = np.concatenate((self.y, [y]))
        wp = np.concatenate((self.weights, [weight]))
        
        # z = A.dot(x)
        
        # A = A - weight*np.outer(z, z)/(1 + weight*x.dot(z))
        
        # b = b + weight*x*y
        
        # theta = A.dot(b)

        W = np.diag(wp)
                
        A = np.eye(self.d)*self.reg + Xp.transpose().dot(W.dot(Xp))
        A = la.pinv(A)
        
        b = Xp.transpose().dot(W.dot(yp))
        
        # (parameter vector)
        theta = A.dot(b)
        
        if keep is True:
            self.theta = theta
            self.A = A
            self.b = b
            # 
            self.weights = wp
            self.X = Xp
            self.y = yp
        
        past_pred = self.X.dot(theta)
        
        pred = theta.dot(x)

        past_mse = self.weights.dot((past_pred - self.y)**2)
        full_mse = past_mse + weight*(x.dot(theta) - y)**2
        
        return (pred, past_mse, full_mse)
    
    def predict(self, X):
        
        assert self.is_fit

        if len(X.shape)==1:
            assert X.shape[0] == self.d
        elif len(X.shape) ==2:
            assert X.shape[1] == self.d
        else:
            assert False, "not implemented"
        
        return X.dot(self.theta)

    def pred_range(self, x, delta):

        assert self.is_fit

        mid = self.theta.dot(x)
        
        half_width = np.sqrt(x.dot(self.Ainv.dot(x)))*np.sqrt(delta)

        return (mid - half_width, mid + half_width)

    def pred_range_coarse(self, x, delta):
        assert self.is_fit

        # mid = self.theta.dot(x)
        
        # half_width = np.linalg.norm(x)*np.sqrt(delta)/self.reg

        # return (mid - half_width, mid + half_width)

        magnitude = 2*self.y_norm/np.sqrt(self.reg)
        return (-magnitude, magnitude)

      
class IncrementalRegressionTree(IncrementalRegressionModel):

    def __init__(self, max_depth = 2):
        
        self.model = tree.DecisionTreeRegressor(max_depth=max_depth)
        
        self.is_fit = False
        
    def get_dataset_size(self):
        
        assert self.is_fit
        
        return self.n
        
    def fit(self, X, y, weights = None):
        
        assert len(X.shape) == 2
        
        self.n = X.shape[0]
        self.d = X.shape[1]
       
        assert self.n == len(y)
       
        self.X = X
        self.y = y

        self.model.fit(X,y)
 
        # Collect additional statistics about the model
        self.leaf_counts = {}

        leaf_indices = self.model.apply(X)
        for leaf in leaf_indices:
            if leaf not in self.leaf_counts:
                self.leaf_counts[leaf]=1
            else:
                self.leaf_counts[leaf]+=1

        # print(self.leaf_counts)
        
        # self.mse =              # TODO
        
        self.is_fit = True
        
        return
        
    def get_mse(self):
        pass
        # assert self.is_fit
        
        # return self.mse
        
    def fit_incremental(self, x, y , weight = 1., keep = False):
        pass

    def fit_incremental_slow(self, x, y , weight = 1., keep = False):
        pass
    
    def predict(self, X):
        
        assert self.is_fit

        if len(X.shape)==1:
            assert X.shape[0] == self.d
        elif len(X.shape) ==2:
            assert X.shape[1] == self.d
        else:
            assert False, "not implemented"
        
        return self.model.predict(X)

    def pred_range(self, x, delta):

        assert self.is_fit

        if len(x.shape) ==1:
            x = x.reshape(1, -1)

        mid = self.model.predict(x)[0]
        
        leaf = self.model.apply(x)[0]
        leaf_count = self.leaf_counts[leaf]

        half_width = np.sqrt(delta/leaf_count)

        return (mid - half_width, mid + half_width)

    def pred_range_coarse(self, x, delta):
        pass
        
class IncrementalRegressionTreeEnsemble(IncrementalRegressionModel):
    
    def __init__(self, n_estimators = 100, max_depth = 2):
        
        self.model = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, max_depth=max_depth)
        
        self.is_fit = False
        
    def get_dataset_size(self):
        
        assert self.is_fit
        
        return self.n
        
    def fit(self, X, y, weights = None):
        
        assert len(X.shape) == 2
        
        self.n = X.shape[0]
        self.d = X.shape[1]
       
        assert self.n == len(y)
       
        self.X = X
        self.y = y

        self.model.fit(X,y)

        self.leaf_counts = []

        for (idx, m) in enumerate(self.model.estimators_):
            # Collect additional statistics about the model
            model = m[0]        # strange indexing
            leaf_counts = {}
            leaf_indices = model.apply(X)
            for leaf in leaf_indices:
                if leaf not in leaf_counts:
                    leaf_counts[leaf]=1
                else:
                    leaf_counts[leaf]+=1
            self.leaf_counts.append(leaf_counts)

        # print(self.leaf_counts)
        
        self.is_fit = True
        
        return
        
    def get_mse(self):
        pass
        # assert self.is_fit
        
        # return self.mse
        
    def fit_incremental(self, x, y , weight = 1., keep = False):
        pass

    def fit_incremental_slow(self, x, y , weight = 1., keep = False):
        pass
    
    def predict(self, X):
        
        assert self.is_fit

        if len(X.shape)==1:
            assert X.shape[0] == self.d
        elif len(X.shape) ==2:
            assert X.shape[1] == self.d
        else:
            assert False, "not implemented"
        
        return self.model.predict(X)

    def pred_range(self, x, delta):

        assert self.is_fit

        if len(x.shape) ==1:
            x = x.reshape(1, -1)

        mid = self.model.predict(x)[0]
        total_half_width = 0.

        for (idx, count) in enumerate(self.leaf_counts):
            est = self.model.estimators_[idx][0]

            leaf = est.apply(x)[0]
            leaf_count = self.leaf_counts[idx][leaf]

            total_half_width +=np.sqrt(delta/leaf_count)

        return (mid - total_half_width, mid + total_half_width)

    def pred_range_coarse(self, x, delta):
        pass

'''
Same as IncrementalRegressionTreeEnsemble, but correctly implements least squares for range prediction.
'''
class IncrementalRegressionTreeEnsemble2(IncrementalRegressionModel):
    
    def __init__(self, n_estimators = 100, max_depth = 2):
        
        self.model = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, max_depth=max_depth)
        
        self.is_fit = False
        
    def get_dataset_size(self):
        
        assert self.is_fit
        
        return self.n
        
    def fit(self, X, y, weights = None):
        
        assert len(X.shape) == 2
        
        self.n = X.shape[0]
        self.d = X.shape[1]
       
        assert self.n == len(y)
       
        self.X = X
        self.y = y

        self.model.fit(X,y)

        '''
        Compute leaf statistics
        '''

        print("Precomputing leaf statistics")

        self.leaf_counts = []
        self.leaf_orders = []

        for (idx, m) in enumerate(self.model.estimators_):
            # Collect additional statistics about the model
            model = m[0]        # strange indexing
            leaf_counts = {}
            leaf_indices = model.apply(X)
            idy = 0
            # TODO replace dict with an array for hashing
            leaf_order = {}
            for (sample_idx, leaf) in enumerate(leaf_indices):
                if leaf not in leaf_counts:
                    leaf_counts[leaf]=1
                    leaf_order[leaf] = idy
                    idy += 1
                else:
                    leaf_counts[leaf]+=1
            self.leaf_counts.append(leaf_counts)
            self.leaf_orders.append(leaf_order)

        self.n_trees = len(self.model.estimators_)

        # dimension of feature space induced by tree ensemble
        self.feature_dim = sum([len(lc) for lc in self.leaf_counts])
        print("Number of ensemble features {}".format(self.feature_dim))

        self.A = np.eye(self.feature_dim, self.feature_dim)

        # TODO: Should only take (#trees)^2 time per sample
        for x in X:
            # feature_vec = self.get_ensemble_features(x.reshape((1,-1)))
            # self.A = self.A + np.outer(feature_vec, feature_vec)
            indices = self.get_ensemble_feature_indices(x.reshape((1,-1)))
            for idx in indices:
                for idy in indices:
                    self.A[idx, idy] += 1
            
        print("Inverting...")
        self.Ainv = la.pinv(self.A)
            
        self.is_fit = True
        
        return

    def get_ensemble_features(self,x):

        feature_vec = np.zeros((1, self.feature_dim))
        for (idx, m) in enumerate(self.model.estimators_):
            leaf_idx = m[0].apply(x)[0]

            feature_vec[0][self.leaf_orders[idx][leaf_idx]]=1
        return feature_vec

    def get_ensemble_feature_indices(self,x):

        # feature_vec = np.zeros((1, self.feature_dim))
        indices = []
        for (idx, m) in enumerate(self.model.estimators_):
            leaf_idx = m[0].apply(x)[0]
            indices.append(self.leaf_orders[idx][leaf_idx])

        return indices

    def get_mse(self):
        pass
        # assert self.is_fit
        
        # return self.mse
        
    def fit_incremental(self, x, y , weight = 1., keep = False):
        pass

    def fit_incremental_slow(self, x, y , weight = 1., keep = False):
        pass
    
    def predict(self, X):
        
        assert self.is_fit

        if len(X.shape)==1:
            assert X.shape[0] == self.d
        elif len(X.shape) ==2:
            assert X.shape[1] == self.d
        else:
            assert False, "not implemented"
        
        return self.model.predict(X)

    def pred_range(self, x, delta):
        '''
        This method should take O((# trees)^2) time to compute, but otherwise be independent of # leaves and #data.
        '''

        assert self.is_fit

        if len(x.shape) ==1:
            x = x.reshape(1, -1)

        mid = self.model.predict(x)[0]

        # Norm squared feature vector under X^(T)X (where X are featurized using tree ensemble)
        val = 0.

        indices = self.get_ensemble_feature_indices(x)
        for idx in indices:
            for idy in indices:

                val += self.Ainv[idx, idy]

            # est = self.model.estimators_[idx][0]

            # leaf = est.apply(x)[0]
            # leaf_count = self.leaf_counts[idx][leaf]

        total_half_width = np.sqrt(val*delta)

        return (mid - total_half_width, mid + total_half_width)

    def pred_range_coarse(self, x, delta):
        pass
    
