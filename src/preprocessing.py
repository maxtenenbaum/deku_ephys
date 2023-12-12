import neo
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def blkrck_to_numpy(filepath):
    reader = neo.io.BlackrockIO(filename=f'{filepath}')
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
        
        # Convert to a NumPy array
        data_array = analog_signal.magnitude
        print('Array shape =', data_array.shape)
        print('Max =', data_array.max(), '// Min =', data_array.min(), '// Mean =', data_array.mean())
        
        return data_array
    else:
        return None

# Example usage
data = blkrck_to_numpy("data/raw/2023_11_20 RefScrewLocation1001.ns6")
print(data)