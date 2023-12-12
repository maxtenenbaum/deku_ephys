import neo
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#%%

def blkrck_to_numpy(filepath, var_name):
    reader = neo.io.BlackrockIO(filename=f'filepath')
    blk = reader.read_block()
    print("Number of segments:", len(blk.segments))
    
    for i, seg in enumerate(blk.segments):
        print("Segment", i)
        print("  Number of analog signals:", len(seg.analogsignals))
        print("  Number of spike trains:", len(seg.spiketrains))
        print("  Number of events:", len(seg.events))
    if len(blk.segments) > 0 and len(blk.segments[0].analogsignals) > 0:
        analog_signal = blk.segments[0].analogsignals[0]
        print(analog_signal)
    # Optionally, convert to a NumPy array and print the shape
    var_name = analog_signal.magnitude
    print('Array shape =',var_name.shape)
    print('Max =',var_name.max(), '// Min =', var_name.min(), '// Mean =', var_name.mean())
    return var_name

#%%

blkrck_to_numpy("/Users/maxtenenbaum/Desktop/Deku Ephys/deku_ephys/data/raw/2023_11_20 RefScrewLocation1001.ns4", "test")
# %%
