from mne.io import read_raw_edf
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import welch, butter, lfilter
from scipy import signal
import pandas as pd
import seaborn as sns
from datetime import datetime
import time
from scipy.interpolate import griddata
from tqdm import tqdm_notebook as tqdm

# Setting
FS = 500                                        #Sampling frequency
LOWCUT = 0.5                                    #Highpass
HIGHCUT = 45                                    #Lowpass
plt.rcParams['figure.figsize'] = [25, 5]

_WINDOW = 15

# Subjects
_DIR_PATH = './gdrive/My Drive/Project_Database'
_SESSION1 = [1528, 1520, 1530, 1507, 1523, 1492, 1517, 1515, 1489, 1503] #1489*
_SESSION2 = [1539, 1541, 1545, 1549, 1551, 1553, 1567, 1569, 1572, 1575]

_STG_PATH = "{_path}/{session}/{session}_alice/csv/STAGE.csv"
_EDF_PATH = "{_path}/{session}/{session}_alice/edf/A000{session}.edf"

# Fucntion to process data

def group_consecutives(vals, step=1):
    # https://stackoverflow.com/questions/7352684/
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

# Calculate Representative Rest1 mean:
def baseline_calc(a):
    e = np.mean(a)
    return a - e

# Banpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut=.5, highcut=45., fs=500, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def psd(s, _fs = FS, _avg='median', fmax = 60):
    """
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html
    
    scaling : { ‘density’, ‘spectrum’ }, optional

        Selects between computing the power spectral density (‘density’) where 
        Pxx has units of V**2/Hz if x is measured in V and computing the power
        spectrum (‘spectrum’) where Pxx has units of V**2 if x is measured in V.
        Defaults to ‘density’.

    """
    _nperseg = 4*_fs
    x, y = welch(s, fs=_fs, average=_avg, nperseg=_nperseg)
    x, y = x[np.where(x<fmax)], y.T[np.where(x<fmax)]

    return x, y 

def GET_PATH(session):
    """
    Return EDF file and STAGE.csv path
    """
    return _EDF_PATH.format_map({"_path":_DIR_PATH, "session":session}),\
    _STG_PATH.format_map({"_path":_DIR_PATH, "session":session})

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class EEG():
    def __init__(self,path):
        self.raw = None
        self.baseline = None
        self.bandpass = None
        self.corrected = None
         
        # Raw data & meta data
        self.path_edf, self.path_stage = path
        raw = read_raw_edf(self.path_edf, preload=True, verbose=0)

        raw.pick_channels(['EEG Fp1-A2','EEG F7-A2','EEG F3-A2','EEG T5-A2', \
                           'EEG O1-A2','EEG Fp2-A1','EEG F4-A1','EEG F8-A1','EEG T6-A1','EEG O2-A1'])
        
        # Rename channel name to standard1005 
        raw.rename_channels({'EEG Fp1-A2': 'Fp1','EEG F7-A2': 'F7',
        'EEG F3-A2': 'F3', 'EEG T5-A2': 'T5','EEG O1-A2': 'O1',
        'EEG Fp2-A1': 'Fp2', 'EEG F4-A1': 'F4', 'EEG F8-A1': 'F8',
        'EEG T6-A1': 'T6', 'EEG O2-A1': 'O2'})
        raw.set_montage('standard_1005', raise_if_subset=False)

        self.raw = raw.copy()

        self.meas_date, _ = self.raw.info['meas_date']
        self.ch_names = self.raw.info['ch_names']
        
        # Stage file: 
        with open(self.path_stage, 'r') as f: 
            stages = f.read().splitlines()
        
            fname, lname, subject, start_date, start_time, \
             end_date, end_time = stages[0].split(',')
        
            stages = stages[1:]

        # Stages and stages indices
        self.stages = stages
        self.subject = subject
        
        task_indices = [idx for idx, _ in enumerate(self.stages) if _ == '13']
        self.tasks = group_consecutives(task_indices)

        rest_indices = [idx for idx, _ in enumerate(self.stages) if _ == '12']
        self.rests = group_consecutives(rest_indices)


        self.start_timestamp = time.mktime(
            datetime.strptime(f"{start_date} {start_time}", \
                              "%m/%d/%Y %I:%M:%S %p").timetuple())
        
        self.DIFFTIME = self.start_timestamp - self.meas_date



    def compute_avg_pxx(self):
        rest_pxx_avg = []
        task_pxx_avg = []
        
        loop = tqdm(self.ch_names, leave=False)

        for ch in loop:
            loop.set_description(ch)
            # Compute pxx for channel ch across chunk of 10s
            pxxs = []
            for task in tqdm(self.tasks, leave=False, desc='Task'):
                for chunk in tqdm(chunks(task, 10), leave=False, desc='Chunk'):
                    _start, _end = self.DIFFTIME + min(chunk), self.DIFFTIME + max(chunk)
                    # Croping task/rest chunk of _WINDOWS second
                    dat = self.raw.copy().crop(_start, _end)
                    # Apply preprocessing step: baseline + bandpass
                    dat.apply_function(baseline_calc)
                    dat.apply_function(butter_bandpass_filter)
                    # Welch PSD
                    sig = dat.copy().pick(ch).get_data()[0:,]
                    f, pxx = psd(sig)
                    pxxs.append(pxx)
            
            pxxs = np.array(pxxs)[:,:,0]
            pxx_avg = np.mean(pxxs, axis=0)
            task_pxx_avg.append(pxx_avg)

            # Compute pxx for channel ch across chunk of 10s
            pxxs = []
            for rest in tqdm(self.rests, leave=False, desc='Rest'):
                for chunk in tqdm(chunks(rest, 10), leave=False, desc='Chunk'):
                    _start, _end = self.DIFFTIME + min(chunk), self.DIFFTIME + max(chunk)
                    # Croping task/rest chunk of _WINDOWS second
                    dat = self.raw.copy().crop(_start, _end)
                    # Apply preprocessing step: baseline + bandpass
                    dat.apply_function(baseline_calc)
                    dat.apply_function(butter_bandpass_filter)
                    # Welch PSD
                    sig = dat.copy().pick(ch).get_data()[0:,]
                    
                    f, pxx = psd(sig)
                    pxxs.append(pxx)
            
            pxxs = np.array(pxxs)[:,:,0]
            pxx_avg = np.mean(pxxs, axis=0)
            rest_pxx_avg.append(pxx_avg)

        return f, rest_pxx_avg, task_pxx_avg


__all__ = ['EEG', 'GET_PATH']