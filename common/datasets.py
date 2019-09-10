"""
This module provides classes for datasets handling with the following 
features:
    - Automatically returns a datasets in the format (data, labels).
    - Supports data splitting in train, test and validation sets.
    - Supports data shuffling.
    - Automatically data enconding and categorical transformation.
    - Data balancing.
Use this module on keras projects.
"""

from inspect import currentframe, getframeinfo
import numpy as np
import collections
import sklearn
import keras
import random
from common.util import logm


def balance_data(x, y):
    """
    Balance data to be representative.

    Note: When balancing the data, the number os data and labels tends to
    decrease because the extra instances of some classes will be removed.        
    
    Args:
        x (list like): Data or paths.
        y (list like): Labels/classes.    

    Returns:
        tuple (np.ndarray, np.ndarray): x, y balanced
    """
    dataset = list(zip(x, y))
    unique, counts = np.unique(y, return_counts=True)
    max_n_instances = min(counts)
    paths_per_label = dict()
    for label in unique:
        paths_per_label[label] = list(filter(lambda d: d[1] == label, dataset))
    x = []
    y = []
    for label in paths_per_label:
        for inst, lbl in paths_per_label[label][:max_n_instances]:
            x.append(inst)
            y.append(lbl)

    return np.asarray(x), np.asarray(y)


class Dataset:
    """
    Dataset class.
    
    Features:
        - Automatically returns a datasets in the format (data, labels).
        - Supports data splitting in train, test and validation sets.
        - Supports data shuffling.
        - Automatically data enconding and categorical transformation.
    """
    def __init__(self,
                 data,
                 labels,
                 name=None,
                 encode_labels=True,
                 to_categorical=True,
                 shuffle=True,
                 num_classes=None,
                 val_split=0.0,
                 test_split=0.0):
        """
        Creates a new dataset with the provided data and labels.
        
        Args:
            data (list like): a list like with data or paths
            labels (list like): a list like with the corresponding labels 
            name (str, optional): The name of this dataset. Defaults to None.
            encode_labels (bool, optional): Choose to encode labels
                automatically. Defaults to True.
            to_categorical (bool, optional): Choose to convert the labels to
                categorical automatically. Defaults to True.
            shuffle (bool, optional): Choose to shuffle the dataset.
                Defaults to True.
            num_classes (int, optional): Number of classes of this dataset.
                Affects the categorical and encoded labels. If None, it will
                be considered the number of different classes in labels.
                Defaults to None.
            val_split (float, optional): Validation split of this dataset.
                Must be in the range of [0.0, 1.0]. Defaults to 0.0.
            test_split (float, optional): Test split of this dataset.
                Must be in the range of [0.0, 1.0]. Defaults to 0.0.
        """
        logm(f'Creating dataset: total of {len(data)} samples',
             cur_frame=currentframe(), mtype='I')
        if len(data) != len(labels):
            logm(f'Size of data ({len(data)}) and labels ({len(labels)}) '
                 'are different', cur_frame=currentframe(), mtype='W')
        if not shuffle and (val_split > 0.0 or test_split > 0.0):
            logm(f'Split is set but no shuffling will be performed',
                 cur_frame=currentframe(), mtype='W')
        if name:
            self._name = name
        self._data = data
        self._labels = labels
        self._shuffle = shuffle
        self._val_split = val_split
        self._test_split = test_split
        self._le = None
        if num_classes is None:
            self._num_classes = np.unique(labels).shape[0]
        else:
            self._num_classes = num_classes

        self._data_labels = list(zip(self._data, self._labels))
        if self._shuffle:
            random.shuffle(self._data_labels)

        all_data = self._data
        all_labels = self._labels
        train_data = list()
        train_labels = list()
        val_data = list()
        val_labels = list()
        test_data = list()
        test_labels = list()

        if self._test_split > 0.0 or self._val_split > 0.0:
            data_per_label = collections.defaultdict(lambda: [])
            for p, l in zip(self._data, self._labels):
                data_per_label[l].append(p)

            splits = [0, 1 - (val_split + test_split), 1 - test_split]

            for label in data_per_label:
                l_train_data = (data_per_label[label][splits[0]:int(splits[1] *
                                                                    len(data_per_label[label]))])
                train_data += l_train_data
                train_labels += [label for _ in range(len(l_train_data))]
                if self._val_split > 0.0:
                    l_val_data = (data_per_label[label][int(splits[1]*len(
                                  data_per_label[label])):int(splits[2]*len(
                                      data_per_label[label]))])
                    val_data += l_val_data
                    val_labels += [label for _ in range(len(l_val_data))]
                if self._test_split > 0.0:
                    l_test_data = (data_per_label[label][int(splits[2]*len(
                                   data_per_label[label])):len(
                        data_per_label[label])])
                    test_data += l_test_data
                    test_labels += [label for _ in range(len(l_test_data))]
        self._all_data = np.asarray(all_data)
        self._all_labels = np.asarray(all_labels)
        self._train_data = np.asarray(train_data)
        self._train_labels = np.asarray(train_labels)
        self._val_data = np.asarray(val_data)
        self._val_labels = np.asarray(val_labels)
        self._test_data = np.asarray(test_data)
        self._test_labels = np.asarray(test_labels)

        if encode_labels:
            self._le = sklearn.preprocessing.LabelEncoder()
            self._le.fit(np.unique(labels))
            self._all_labels = self._le.transform(self._all_labels)
            self._train_labels = self._le.transform(self._train_labels)
            self._test_labels = self._le.transform(self._test_labels)
            self._val_labels = self._le.transform(self._val_labels)
        if to_categorical:
            self._all_labels = keras.utils.to_categorical(
                self._all_labels, num_classes=self._num_classes)
            self._train_labels = keras.utils.to_categorical(
                self._train_labels, num_classes=self._num_classes)
            self._test_labels = keras.utils.to_categorical(
                self._test_labels, num_classes=self._num_classes)
            self._val_labels = keras.utils.to_categorical(
                self._val_labels, num_classes=self._num_classes)

    def __call__(self) -> (np.ndarray, np.ndarray):
        """
        The calling of a object of this class will return all the dataset,
        including all sets (validation, test and train sets).
        The data and labels returned are transformed by the transformations
        provided in the init method, that is, encoding and shuffling for
        example.
        
        Returns:
            tuple: (data, labels)
        """
        return self._all_data, self._all_labels

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes of this dataset.
        
        Returns:
            int: number of classes
        """
        return self._num_classes

    @property
    def train(self) -> (np.ndarray, np.ndarray):
        """
        Returns the train data.
        
        Returns:
            (np.ndarray, np.ndarray): (data, labels)
        """
        return self._train_data, self._train_labels

    @property
    def test(self) -> (np.ndarray, np.ndarray):
        """
        Returns the test data. It can be None if no test split was provided
        in the init method.
        
        Returns:
            (np.ndarray, np.ndarray): (data, labels)
        """
        return self._test_data, self._test_labels

    @property
    def validation(self) -> (np.ndarray, np.ndarray):
        """
        Returns the validation data. It can be None if no test split was
        provided in the init method.
        
        Returns:
            (np.ndarray, np.ndarray): (data, labels)
        """
        return self._val_data, self._val_labels

    @property
    def label_encoder(self) -> sklearn.preprocessing.LabelEncoder:
        """
        Get the label encoder object for labels transformations.
        This is useful to get the original labels.
        
        Returns:
            sklearn.preprocessing.LabelEncoder: The label encoder object
        """
        return self._le

    def __str__(self):
        """
        Returns the name of the dataset or call the super class method.
        """
        if self._name:
            return self._name
        return super().__str__()

    def __len__(self):
        """
        Returns the size of all data.
        """
        return len(self._all_data)

class TestDataset(Dataset):
    """
    A more specific type of Dataset used for a specific test set."

    Calling the test() method of this object will return all data as the test
    set. It is not possible to split the data.
    """
    def __init__(self,
                 data,
                 labels,
                 name=None,
                 num_classes=None,
                 encode_labels=True,
                 to_categorical=True,
                 shuffle=True):
        """
        Creates a new dataset with the provided data and labels.
        
        Args:
            data (list like): a list like with data or paths
            labels (list like): a list like with the corresponding labels 
            name (str, optional): The name of this dataset. Defaults to None.
            encode_labels (bool, optional): Choose to encode labels
                automatically. Defaults to True.
            to_categorical (bool, optional): Choose to convert the labels to
                categorical automatically. Defaults to True.
            shuffle (bool, optional): Choose to shuffle the dataset.
                Defaults to True.
            num_classes (int, optional): Number of classes of this dataset.
                Affects the categorical and encoded labels. If None, it will
                be considered the number of different classes in labels.
                Defaults to None.
            val_split (float, optional): Validation split of this dataset.
                Must be in the range of [0.0, 1.0]. Defaults to 0.0.
            test_split (float, optional): Test split of this dataset.
                Must be in the range of [0.0, 1.0]. Defaults to 0.0.
        """
        Dataset.__init__(self,
                         data=data,
                         labels=labels,
                         name=name,
                         num_classes=num_classes,
                         encode_labels=encode_labels,
                         to_categorical=to_categorical,
                         shuffle=shuffle)

    @property
    def test(self) -> (np.ndarray, np.ndarray):
        """
        Returns the test data. 
        
        Returns:
            (np.ndarray, np.ndarray): (data, labels)
        """
        return super().__call__()

    def __str__(self):
        return super().__str__() + '[TestDataset]'