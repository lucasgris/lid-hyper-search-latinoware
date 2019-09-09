"""
This module provides generator and utilities functions for batching.
The generator is a keras sequence instance. 
You can use the heap to improve performance in case the CPU usage overloaded 
with data loading.
"""

import numpy as np
import hashlib
import glob
import os
import threading
from common.util import logm
import shutil
import random
from inspect import currentframe, getframeinfo
from keras.utils import Sequence


class Heap:
    """
    Heap for data caching.
    """
    def __init__(self,
                 save_fn=np.save,
                 load_fn=np.load,
                 heap_dir='tmp',
                 file_format='npy',
                 recover_state=True,
                 max_size=None,
                 use_item_as_key=False,
                 depth=4):
        """
        Creates a heap.
        
        Args:
            save_fn (callable, optional): Function to save data.
                Must be the following 
                Defaults to np.save.
            load_fn (callable, optional): Function to load data.
                Defaults to np.load.
            heap_dir (str, optional): Heap directory. Defaults to 'tmp'.
            file_format (str, optional): . Defaults to 'npy'.
            recover_state (bool, optional): Choose to recover a previous 
                state of the heap. This will force the heap to get all saved
                files in the provided directory. Defaults to True.
            max_size (int, optional): Max size of the heap. When the heap
                reaches the max size, it will start to delete random instances
                automatically. Defaults to None.
            use_item_as_key (bool, optional): Use the str(item) as key not the
                name. Use carefully. Defaults to False.
            depth (int, optional): Depth of the heap. More depth means more
                subdirectories. Defaults to 4.
        """
        logm(f'Creating heap at {heap_dir}', cur_frame=currentframe(),
             mtype='I')
        os.makedirs(heap_dir, exist_ok=True)
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.files = set()
        self.depth = depth
        self.heap_dir = heap_dir
        self.file_format = file_format
        self.use_item_as_key = use_item_as_key
        self.max_size = max_size
        if recover_state:
            saved_files = glob.glob(self.heap_dir + '/**/*.' +
                                    self.file_format,
                                    recursive=True)
            self.files = set([os.path.basename(fp) for fp in saved_files])
            logm(f'Recovering state of heap at {heap_dir}:'
                 f' Found {len(self.files)} files',
                 cur_frame=currentframe(),
                 mtype='I')

    def _generate_file_path(self, key):
        hs = int(hashlib.sha256(str(key).encode('utf-8')).hexdigest(), 16)
        path = ''
        step = 2
        for i in range(0, self.depth + 1, step):
            path = os.path.join(path, str(hs)[i: i + step])
        return path

    def _delete_random(self, n_files):
        files = random.sample(self.files, n_files)
        for file in files:
            logm(f'Removing file {file}',
                 cur_frame=currentframe(),
                 mtype='I',
                 stdout=False)
            self.files.remove(file)
            os.remove(file)

    def add(self, item, key, replace_if_exists=False):
        """
        Adds a new item.
        
        Args:
            item: The item to save.
            key (str): Key of the file. Usually its name.
            replace_if_exists (bool, optional): It True, will replace existing
                files with the same key. Defaults to False.
        """
        f_path = self._generate_file_path(item if self.use_item_as_key else
                                          key)
        os.makedirs(os.path.join(self.heap_dir, f_path), exist_ok=True)
        try:
            self.save_fn(os.path.join(self.heap_dir, f_path, key), item)
            self.files.add(f'{key}.{self.file_format}')
        except FileExistsError as fe:
            logm(f'File exists ({str(fe)})', cur_frame=currentframe(),
                 mtype='W')
            if replace_if_exists:
                logm(f'Replacing file {key}', cur_frame=currentframe(),
                     mtype='W')
                os.remove(os.path.join(self.heap_dir, f_path,
                                       f'{key}.{self.file_format}'))
                self.save_fn(os.path.join(self.heap_dir, f_path, key), item)
                self.files.add(f'{key}.{self.file_format}')
        if self.max_size:
            files = glob.glob(self.heap_dir + '/**/*.' + self.file_format,
                              recursive=True)
            if len(files) > self.max_size:
                logm(f'Max heap size {self.max_size} '
                     'reached: removing 100 random files',
                     cur_frame=currentframe(), mtype='W')
                threading.Thread(target=self._delete_random,
                                 args=100).start()

    def get(self, key):
        """
        Gets the file with the provided key.
        
        Args:
            key (str): Key of the file.
        
        Returns:
            obj: Object based on the load_fn of this object.
        """
        f_path = self._generate_file_path(key)
        return self.load_fn(os.path.join(self.heap_dir, f_path,
                                         f'{key}.{self.file_format}'))

    def __contains__(self, key):
        return f'{key}.{self.file_format}' in self.files

    def destruct(self):
        """
        Destruct heap and remove all files.
        """
        self.files = set()
        shutil.rmtree(self.heap_dir)


class Generator(Sequence):
    """
    Generator for data batching. See keras.utils.Sequence.
    """
    def __init__(self, paths, labels, batch_size: int, loader_fn: callable,
                 shuffle: bool = True, expected_shape=None, loader_kw=None,
                 heap=None, not_found_ok=False):
        """
        Creates a new generator.
        
        Args:
            paths (list like): List containing paths.
            labels (list like): List of the respective data labels.
            batch_size (int): Batch size.
            loader_fn (callable): Function for data loading.
            shuffle (bool, optional): If True, will shuffle the data before.
                Defaults to True.
            expected_shape (tuple, optional): If not None, it will check each
                shape of the data loaded. Defaults to None.
            loader_kw (dict, optional): Key arguments to pass on to the loader.
                Defaults to None.
            heap (Heap, optional): Heap instance. Defaults to None.
            not_found_ok (bool, optional): Choose to load another instance if
                the loader fails to find a file. Defaults to False.
        """
        assert len(paths) > 0
        self._paths = paths
        self._labels = labels
        self._batch_size = batch_size
        self._loaderkw = loader_kw if loader_kw else {}
        self._loader = loader_fn
        self._expected_shape = expected_shape
        self._heap = heap
        self._not_found_ok = not_found_ok

        if shuffle:
            dataset = list(zip(self._paths, self._labels))
            random.shuffle(dataset)
            self._paths, self._labels = zip(*dataset)

    def _get_random_instance(self):
        i = np.random.randint(0, len(self._paths))
        return self._paths[i], self._labels[i]

    def __getitem__(self, index) -> (np.ndarray, np.ndarray):
        paths = self._paths[(index*self._batch_size):
                            ((index+1)*self._batch_size)]
        labels = self._labels[(index*self._batch_size):
                              ((index+1)*self._batch_size)]
        paths_and_labels = list(zip(paths, labels))
        # Fill batches
        x = []
        y = []
        threshold = 0
        for path_label in paths_and_labels:
            if self._not_found_ok:
                try:
                    # TODO: check if is more optimal load from heap or not
                    # TODO: duplicated code
                    if self._heap and os.path.basename(path_label[0]) in \
                            self._heap:
                        x.append(self._heap.get(os.path.basename(
                            path_label[0])))
                    else:
                        data = self._loader(path_label[0], **self._loaderkw)
                        if self._heap:
                            self._heap.add(data, os.path.basename(
                                path_label[0]))
                        x.append(data)
                    y.append(path_label[1])
                except FileNotFoundError as fnf:
                    logm(f'File {path_label[0]} not found ({str(fnf)})',
                         cur_frame=currentframe(),
                         mtype='E')
                    # If not found, append a new path to load
                    p, l = self._get_random_instance()
                    paths_and_labels.append((p, l))
                    # Increase a threshold value to avoid infinite loops
                    threshold += 1

                    if threshold == 10:
                        # (threshold can be any value)
                        raise RuntimeError(
                            'Threshold value reached. Error when '
                            'trying to read the files provided '
                            '(not able to fill the batch).')
                    continue
            else:  # Read data without handling the exception
                # TODO: duplicated code
                if (self._heap and os.path.basename(path_label[0]) in
                        self._heap):
                    x.append(self._heap.get(os.path.basename(
                        path_label[0])))
                else:
                    data = self._loader(path_label[0], **self._loaderkw)
                    if self._heap:
                        self._heap.add(data, os.path.basename(
                            path_label[0]))
                    x.append(data)
                y.append(path_label[1])

            if (self._expected_shape is not None and x[-1].shape !=
                    self._expected_shape):
                logm(f'Expected shape {self._expected_shape} when loading '
                     f'{path_label[0]}. But found shape of {x[-1].shape} '
                     'instead', cur_frame=currentframe(), mtype='W')
                # TODO: remove file
                # If the last read data is not in the expected shape
                p, l = self._get_random_instance()
                paths_and_labels.append((p, l))
                # Increase a threshold value to avoid infinite loops
                threshold += 1
                # Remove the last instance
                x.pop()
                y.pop()

                # If all data was tried to be read, raise an exception
                if threshold == self._batch_size:
                    err = RuntimeError('Threshold value reached. Error when '
                                       'trying to read the files provided '
                                       '(not able to fill the batch).')
                    logm(f'Exception {str(err)}',
                         cur_frame=currentframe(),
                         mtype='E')
                    raise err
                continue
        return np.asarray(x), np.asarray(y)

    def _getitem(self, index) -> (np.ndarray, np.ndarray):
        paths = self._paths[(index*self._batch_size):
                            ((index+1)*self._batch_size)]
        labels = self._labels[(index*self._batch_size):
                              ((index+1)*self._batch_size)]
        paths_and_labels = list(zip(paths, labels))
        # Fill batches
        x = []
        y = []
        for path_label in paths_and_labels:
            if self._heap and os.path.basename(path_label[0]) in self._heap:
                x.append(self._heap.get(os.path.basename(path_label[0])))
            else:
                data = self._loader(path_label[0], **self._loaderkw)
                if self._heap:
                    self._heap.add(data, os.path.basename(path_label[0]))
                x.append(data)
            y.append(path_label[1])
        return np.asarray(x), np.asarray(y)

    def __len__(self):
        return int(np.floor(len(self._paths) / self._batch_size))