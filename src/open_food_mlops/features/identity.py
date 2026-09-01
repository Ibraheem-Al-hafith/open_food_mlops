"""
Identity Feature Transforemer, used for testing the features pipeline
Classes:
    IdentityTransformer :
        It just pass the data frame as it.
"""
import pandas as pd
import numpy as np
from .base import BaseFeatureTransformer

class IdentityTransformer(BaseFeatureTransformer):
    """Calculates the character length of a text column."""
    
    def __init__(self):
        # 1. ALWAYS call super().__init__() to set up base tracking
        super().__init__()
        
    def _fit_transformer(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """
        Learn state from training data. 
        Leave this empty for stateless transformers!
        """
        pass

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation. 
        MUST return a pandas DataFrame.
        """
        # Pro-tip: Use .copy() to avoid mutating the original DataFrame!
        X_out = X.copy()
        X_out = X_out.replace([np.inf, -np.inf], np.nan)
        X_out = X_out.fillna(0)
        
        return X_out