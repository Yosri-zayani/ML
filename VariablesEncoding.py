# column encoder (easier ways to encode data ) 
import pandas as pd
import numpy as np

data = [
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2'],
    ['green', 'S', 8.7, 'class3'],
    ['red', 'M', 11.2, 'class1'],
    ['blue', 'L', 12.4, 'class2'],
    ['green', 'XL', 16.5, 'class1'],
    ['red', 'S', 9.1, 'class3'],
    ['blue', 'M', 10.9, 'class2']
]

# Create DataFrame with named columns
columns = ['color', 'size', 'value', 'class']
dataset = pd.DataFrame(data, columns=columns)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0]),
        ('ordinal', OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']]), [1])
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

X
