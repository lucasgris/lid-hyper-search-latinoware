from sklearn.preprocessing import MinMaxScaler
from scipy.signal import spectrogram
from inspect import currentframe, getframeinfo
from PIL import Image
import numpy as np
import librosa
import os
import cv2

from common.config import SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH
from common.config import SAMPLING_RATE
from common.util import logm


def normalize(data):
    """
    Normalizes the spectrogram data. This will scale the data to the range
    [0, 1].
    
    Args:
        data (np.ndarray): Data spectrogram to normalize.
    
    Returns:
        np.ndarray: The normalized data.
    """
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 255))
    ascolumns = data.reshape(-1, 3)
    t = scaler.fit_transform(ascolumns)
    transformed = t.reshape(data.shape)
    data = transformed.astype('float32')
    data /= 255.0
    return data


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    """
    Compute and returns a log spectrogram of an audio.
    
    Args:
        audio (np.ndarray): Audio data. Must read audio first.
        sample_rate (int): Sample rate of the audio.
        window_size (int, optional): Window size. Defaults to 20.
        step_size (int, optional): Step size. Defaults to 10.
        eps (float, optional): Eps. Defaults to 1e-10.
    
    Returns:
        Computed spectrogram: [description]
    """
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = spectrogram(audio,
                                     fs=sample_rate,
                                     window='hann',
                                     nperseg=nperseg,
                                     noverlap=noverlap,
                                     detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def spec_save(out, spec, fmt='png'):
    """
    Save the spectrogram as image.
    
    Args:
        out (str): output path.
        spec (np.ndarray):  spectrogram data.
        fmt (str, optional): The image format. Defaults to 'png'.
    """
    # TODO: use numpy arrays
    # Might be not necessary to use 3 channels of data to train the model
    # since the image is in gray scale. But, due to the image conversion
    # applied using the cv2.cv2Color method, the generated data has the
    # 3 channels. May be interesting using other format of data, for example
    # numpy arrays generated directly by the spectrogram function.
    # The conversion was performed for the PIL.Image.fromarray loading to work
    # properly.
    result = Image.fromarray((spec * 255.0).astype(np.uint8))
    result.save(out + '.' + fmt)


def spec_load_and_rshp(path, expected_fmt='png', remove_bad_file=True):
    """
    Loads an image and reshapes in the proper format.
    
    Args:
        path (str): path to the image. 
        expected_fmt (str, optional): Expected format of the data.
            Choices = [png, npy]. If png will load as image using Image.open,
            if npy will load as numpy array using np.load. Defaults to 'png'.
        remove_bad_file (bool, optional): If True will remove the file if it
            is not in the expected shape or format. Defaults to True.
    
    Raises:
        err (ValueError): If the file is not in the expected format. 
    
    Returns:
        np.ndarray: An array of shape (SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, 3)
            with the loaded data or zeros if an expection occured when
            loading the data.
    """
    if path[-3:] != expected_fmt:
        err = ValueError(f'File must be in format {expected_fmt}. '
                         f'(When trying to load {path})')

        logm(f'Exception {str(err)}',
             cur_frame=currentframe(),
             mtype='E')
        raise err
    try:
        if format == 'png':
            spc = np.array(Image.open(path)).reshape(SPEC_SHAPE_HEIGTH,
                                                     SPEC_SHAPE_WIDTH,
                                                     3)
        if format == 'npy':
             spc = np.load(path).reshape(SPEC_SHAPE_HEIGTH,
                                         SPEC_SHAPE_WIDTH,
                                         3)
        spc = normalize(spc)
    except Exception as ex:
        logm(f'Bad file: {str(ex)} (when trying to load {path})',
             cur_frame=currentframe(), mtype='E')
        if remove_bad_file:
            logm(f'Removing file {path}', cur_frame=currentframe(),
                 mtype='I')
            if not os.path.isfile(path):
                logm(f'Removing file {path}: is not a file',
                     cur_frame=currentframe(), mtype='W')
            else:
                os.remove(path)
        return np.zeros(shape=(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, 3))
    return spc


def wav_to_specdata(path,
                    convert_to_rgb=True,
                    normalize_pixels=True,
                    duration=5):
    """
    Reads an wav file and returns the computed spectrogram.
    
    Args:
        path (str): path to the audio file (wav).
        convert_to_rgb (bool, optional): If True will convert the spectrogram
            data to rgb (3 channels). Defaults to True.
        normalize_pixels (bool, optional): If True will normalize the data
            in the range [0, 1]. See normalize. Defaults to True.
        duration (int, optional): Duration to load the data. Defaults to 5.
    
    Returns:
        np.ndarray: Spectrogram data ready to feed the model.
    """
    b, sr = librosa.load(path, duration=duration, sr=SAMPLING_RATE)
    _, _, spc = log_specgram(b, sr)
    spc = spc.astype('float32')
    spc = np.rot90(spc)
    spc = spc.reshape(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH)
    spc = spc.reshape(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, 1)
    if convert_to_rgb:
        spc = cv2.cvtColor(spc, cv2.COLOR_GRAY2RGB)
        spc = spc.reshape(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, 3)
    if normalize_pixels:
        spc = normalize(spc)
    return spc
