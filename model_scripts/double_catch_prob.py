import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class catch_prob(BaseEstimator, ClassifierMixin):
    def __init__(self,normal,wall):
        self.normal = normal
        self.wall = wall

    def predict_proba(self,X):
        p = np.zeros((len(X),2))
        wall_ball_mask = X[:,-1]==1
        X_wall = X[ wall_ball_mask,:-1]
        X_norm = X[~wall_ball_mask,:6]
        p_wall = self.wall.predict_proba(X_wall)[:,-1]
        p_norm = self.normal.predict_proba(X_norm)[:,-1]
        p[ wall_ball_mask,1] = p_wall
        p[~wall_ball_mask,1] = p_norm
        p[:,0] = 1-p[:,1]
        return p

