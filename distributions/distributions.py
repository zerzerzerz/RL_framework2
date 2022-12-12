raise NotImplementedError
from torch import distributions as D

class Categorical(D.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)
    
    def sample(self):
        '''return [1,1]'''
        return super().sample().unsqueeze(0)


class MultivariateNormal(D.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
    
    def sample(self):
        '''return [1,d]'''
        return super().sample().unsqueeze(0)