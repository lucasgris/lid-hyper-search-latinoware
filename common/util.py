from datetime import datetime
from inspect import currentframe, getframeinfo
from common.config import LOG_FILE
import subprocess
import sys
import os
import time


def logm(message, mtype='I', cur_frame=None, stdout=True):
    """
    Basic logging function. It shows the log in the format:
    datetime.now: [mtype]: [current frame]: [message]
    
    Args:
        message (str): [description]
        mtype (str, optional): Message type. Usually:
            - I, for information;
            - W, for warnings;
            - E, for errors.
            Defaults to 'I'.
        cur_frame (inspect.currentframe, optional): Current frame.
            Defaults to None.
        stdout (bool, optional): Choose to show the log in stdout.
            Defaults to True.
    """
    loc = (f'{cur_frame.f_code.co_filename}:{cur_frame.f_lineno}'
           if cur_frame else '')
    log_m = f'{datetime.now()}: {mtype} {loc}: {message}'
    with open(LOG_FILE, 'a') as out:
        out.write(f'{log_m}\n')
    if stdout:
        print(log_m)


class Timer:
    """
    Context class for timing.

    Usage:
        with Timer() as t:
            # do stuff
        print('Total time taken:', t.interval)
    """
    def __enter__(self):
        self._start = time.time()
        self._interval = None
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self._interval = self._end - self._start

    @property
    def interval(self):
        return self._interval


def parse_csv(path: str,
              prefix_path: str = None,
              remove_path: str = None) -> (list, list):
    """
    Parses a file containing datasets paths and labels.

    Expected format:
        'name_of_file_1.ext', ..., ..., 'label_1'
        'name_of_file_2.ext', ..., ..., 'label_2'
        ...
        'name_of_file_n.ext', ..., ..., 'label_n'

        *ext is the file extension.
    
    Args:
        path (str): path to the csv file.
        prefix_path (str, optional): path to add in the beginning of each path
            data. Defaults to None.
        remove_path (str, optional): path to remove from the beginning of each
            path data. Defaults to None.
    
    Returns:
        tuple (list, list): A tuple with (pahts, labels).
    
    Notes:
        Characters like `\` or `/` will be replaced by os.sep.
        prefix_path will be add at the beginning of each path. 
        remove_path will be removed from the beginning of each path.
        remove_path runs first.
    """
    path = path
    file_paths = []
    labels = []
    with open(path) as dataset:
        for line in dataset:
            data = line.split(',')
            p = data[0]
            if os.sep == '\\':
                p = p.replace('/', os.sep)
            elif os.sep == '/':
                p = p.replace('\\', os.sep)
            if remove_path is not None:
                p = str(p.split(remove_path)[-1])
            if prefix_path is not None:
                file_paths.append(os.path.join(prefix_path, p))
            else:
                file_paths.append(p)
            labels.append(data[-1].rstrip())
    return file_paths, labels


def syscommand(command: str, debug: bool = False):
    """
    Runs a system command

    debug: bool
        If True, will print the command.
    return_out: bool
    """
    if debug:
        logm(command, cur_frame=currentframe())
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True)
    stdout, stderr = process.communicate()
    if debug and stderr is not None:
        logm(stderr.decode(sys.stdout.encoding), mtype='E',
             cur_frame=currentframe())
    return stdout.decode(sys.stdout.encoding)
