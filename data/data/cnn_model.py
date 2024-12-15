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
    """
    Handles the conversion of surnames into numerical vectors that can be processed by the neural network
    Creates one-hot encoded matrices for the input surnames.
    """
    def __init__(self,surname_vocab,nationality_vocab):
        #Stores vocabularies for both surnames(characters) and nationalities
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self,surname):
        """
        Convert a surname string into one-hot encoded matrix
        Each column represents a character position
        Each row represents a possible character in the vocabulary
        """
        vocab_size = len(self.surname_vocab)
        max_surname_length = 20 #Maximum length of surname to process

        #Create empty matrix of zeros
        surname_vector = np.zeros((vocab_size,max_surname_length), dtype= np.float32)

        #Fill matrix with 1s at appropriate positions
        for char_index, character in enumerate(surname[:max_surname_length]):
            vocab_index = self.surname_vocab.lookup_token(character)
            surname_vector[vocab_index][char_index] = 1
        return surname_vector
    
    @classmethod
    def from_dataframe(cls,surname_df):
        """
        Create a vectorizer from a DataFrame containing surnames and nationalities
        Builds vocabularies for both characters in surnames and nationality labels
        """
        surname_vocab = Vocabulary()
        nationality_vocab = Vocabulary()

        #Add unknown token for handling unseens characters
        surname_vocab.add_token('<UNK>')

        #Build character vocabulary for surnames
        for surname in surname_df.surname:
            for characters in surname:
                surname_vocab.add_token(characters)
        
        #Build nationality vocabulary
        for nationality in surname_df.nationality:
            nationality_vocab.add_token(nationality)
        return cls(surname_vocab,nationality_vocab)
    
    