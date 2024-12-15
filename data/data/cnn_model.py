import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import string

class Vocabulary:
    """
    A class to handle text vocabulary and conversion between token and indices
    """
    def __init__(self,token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx:token for token,idx in self._token_to_idx.items()}
    
    def add_token(self,token):
        """
        Add a new token to the Vocabulary if it doesn't exist
        Each token gets a unique index based on the current vocabulary size
        """
        if token not in self._token_to_idx:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._token_to_idx[index] = token
    
    def lookup_token(self,token):
        """
        Convert a token to it's corresponding index
        Return the UNK token (unknown) token index if token is not previously existing
        """
        return self._token_to_idx[token]
    
    def lookup_index(self,index):
        """
        Convert a index back to it's corresponding token
        Used when interpreting model predictions
        """
        return self._idx_to_token[index]
    

class SurnameVectorizer:
    
    