''' 
    Adapted from
        - https://github.com/zzzace2000/nodegam/blob/main/nodegam/data.py
        - https://github.com/google-research/google-research/blob/master/neural_additive_models/data_utils.py
'''

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

class OurDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)

class Preprocessor: ### adopted from NODE-GAM (Chang et al., 2022)
    def __init__(self, random_state=0, cat_features=None,
                 y_normalize=False, quantile_transform=False,
                 output_distribution='normal', n_quantiles=2000,
                 quantile_noise=1e-3):
        """Preprocessor does the data preprocessing like input and target normalization.

        Args:
            random_state: Global random seed for an experiment.
            cat_features: If passed in, it does the ordinal encoding for these features before other
                input normalization like quantile transformation. Default: None.
            y_normalize: If True, it standardizes the targets y by setting the mean and stdev to 0
                and 1. Useful in the regression setting.
            quantile_transform: If True, transforms the features to follow a normal or uniform
                distribution.
            output_distribution: Choose between ['normal', 'uniform']. Data is projected onto this
                distribution. See the same param of sklearn QuantileTransformer. 'normal' is better.
            n_quantiles: Number of quantiles to estimate the distribution. Default: 2000.
            quantile_noise: If specified, fits QuantileTransformer on data with added gaussian noise
                with std = :quantile_noise: * data.std; this will cause discrete values to be more
                separable. Please note that this transformation does NOT apply gaussian noise to the
                resulting data, the noise is only applied for QuantileTransformer.

        Example:
            >>> preprocessor = Preprocessor(
            >>>     cat_features=['ethnicity', 'gender'],
            >>>     y_normalize=True,
            >>>     random_state=1337,
            >>> )
            >>> preprocessor.fit(X_train, y_train)
            >>> X_train, y_train = preprocessor.transform(X_train, y_train)
        """

        self.random_state = random_state
        self.cat_features = cat_features
        self.y_normalize = y_normalize
        self.quantile_transform = quantile_transform
        self.output_distribution = output_distribution
        self.quantile_noise = quantile_noise
        self.n_quantiles = n_quantiles

        self.transformers = []
        self.y_mu, self.y_std = 0, 1
        self.feature_names = None

    def fit(self, X, y):
        """Fit the transformer.

        Args:
            X (pandas daraframe): Input data.
            y (numpy array): target y.
        """
        assert isinstance(X, pd.DataFrame), 'X is not a dataframe! %s' % type(X)
        self.feature_names = X.columns

        if self.cat_features is not None:
            cat_encoder = LeaveOneOutEncoder(cols=self.cat_features)
            cat_encoder.fit(X, y)
            self.transformers.append(cat_encoder)

        if self.quantile_transform:
            quantile_train = X.copy()
            if self.cat_features is not None:
                quantile_train = cat_encoder.transform(quantile_train)

            if self.quantile_noise:
                r = np.random.RandomState(self.random_state)
                stds = np.std(quantile_train.values, axis=0, keepdims=True)
                noise_std = self.quantile_noise / np.maximum(stds, self.quantile_noise)
                quantile_train += noise_std * r.randn(*quantile_train.shape)

            qt = QuantileTransformer(random_state=self.random_state,
                                     n_quantiles=self.n_quantiles,
                                     output_distribution=self.output_distribution,
                                     copy=False)
            qt.fit(quantile_train)
            self.transformers.append(qt)

        if y is not None and self.y_normalize:
            self.y_mu, self.y_std = y.mean(axis=0), y.std(axis=0)
            print("Normalize y. mean = {}, std = {}".format(self.y_mu, self.y_std))

    def transform(self, *args):
        """Transform the data.

        Args:
            X (pandas daraframe): Input data.
            y (numpy array): Optional. If passed in, it will do target normalization.

        Returns:
            X (pandas daraframe): Normalized Input data.
            y (numpy array): Optional. Normalized y.
        """
        assert len(args) <= 2

        X = args[0]
        if len(self.transformers) > 0:
            X = X.copy()
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)

            for i, t in enumerate(self.transformers):
                # Leave one out transform when it's training set
                X = t.transform(X)

        # Make everything as numpy and float32
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float32)

        if len(args) == 1:
            return X

        y = args[1]
        if y is None:
            return X, None

        if self.y_normalize and self.y_mu is not None and self.y_std is not None:
            y = (y - self.y_mu) / self.y_std
            y = y.astype(np.float32)

        return X, y

def load_mimic2(DATA_PATH = "./data", fold=0):
    print(os.getcwd())
    cols = ['Age', 'GCS', 'SBP', 'HR', 'Temperature',
            'PFratio', 'Renal', 'Urea', 'WBC', 'CO2', 'Na', 'K',
            'Bilirubin', 'AdmissionType', 'AIDS',
            'MetastaticCancer', 'Lymphoma', 'HospitalMortality']
    
    df = pd.read_csv(os.path.join(DATA_PATH, 'mimic2/mimic2.data'), names=cols, delim_whitespace=True)
    
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1].values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic2', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic2', 'test%d.txt') % fold, header=None)[0].values

    cat_features = ['GCS', 'Temperature', 'AdmissionType', 'AIDS',
                    'MetastaticCancer', 'Lymphoma', 'Renal']
    for c in cat_features:
        X_df[c] = X_df[c].astype('string')

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx],
        'cat_features': cat_features
    }

def load_mimic3(DATA_PATH = "./data", fold=0):

    df = pd.read_csv(os.path.join(DATA_PATH, 'mimic3/adult_icu.gz'), compression='gzip')

    train_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian',
        'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max',
        'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min',
        'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate',
        'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']

    label = 'mort_icu'

    X_df = df[train_cols]
    y_df = df[label].values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic3', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'mimic3', 'test%d.txt') % fold, header=None)[0].values

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx]
    }

def load_income(DATA_PATH = "./data", fold=0):
    '''Adult Income dataset'''
    cols = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    df = pd.read_csv(os.path.join(DATA_PATH, 'adult/adult.data'), header=None)
    df.columns = cols

    X_df = df.iloc[:, :-1]

    y_df = df.iloc[:, -1].copy()
    # Make it as 0 or 1
    y_df.loc[y_df == ' >50K'] = 1.
    y_df.loc[y_df == ' <=50K'] = 0.
    y_df = y_df.values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'adult', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'adult', 'test%d.txt') % fold, header=None)[0].values

    cat_features = X_df.columns[X_df.dtypes == object]

    for c in cat_features:
        X_df[c] = X_df[c].astype('string')

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx],
        'cat_features': cat_features
    }

def load_credit(DATA_PATH = "./data", fold=0):

    df = pd.read_csv(os.path.join(DATA_PATH, 'credit/credit/creditcard.csv'))

    df = df.dropna()
    X_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]
    y_df = y_df.values.astype(np.int32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, 'credit/credit', 'train%d.txt') % fold, header=None)[0].values
    test_idx = pd.read_csv(os.path.join(DATA_PATH, 'credit/credit', 'test%d.txt') % fold, header=None)[0].values

    return {
        'problem': 'classification',
        'X_train': X_df.iloc[train_idx],
        'y_train': y_df[train_idx],
        'X_test': X_df.iloc[test_idx],
        'y_test': y_df[test_idx]
    }

def load_housing(fold=0):

    data = fetch_california_housing()
    X = data.data.copy()
    y = data.target.copy()
    feature_names = data.feature_names

    X = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=0,
        shuffle=True,
        stratify=None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1 / (0.1 + 0.7),
        random_state=0,
        shuffle=True,
        stratify=None
    )
    
    return {
        'problem': 'regression',
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }

def load_year(DATA_PATH = "./data", test_size=51630):

    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int32) for i in range(n_features)}
    data = pd.read_csv(os.path.join(DATA_PATH, "year", "YearPredictionMSD.txt"), header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:], data_train.iloc[:, 0].values.astype(np.float32)
    X_test, y_test = data_test.iloc[:, 1:], data_test.iloc[:, 0].values.astype(np.float32)

    train_idx = pd.read_csv(os.path.join(DATA_PATH, "year", 'stratified_train_idx.txt'), header=None)[0].values
    valid_idx = pd.read_csv(os.path.join(DATA_PATH, "year", 'stratified_valid_idx.txt'), header=None)[0].values
        
    return {
        'problem': 'regression',
        'X_train': X_train.iloc[train_idx],
        'y_train': y_train[train_idx],
        'X_valid': X_train.iloc[valid_idx],
        'y_valid': y_train[valid_idx],
        'X_test': X_test,
        'y_test': y_test
    }
