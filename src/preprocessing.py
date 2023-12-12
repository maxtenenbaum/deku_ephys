import neo
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#%%

def blkrck_to_numpy(filepath):
    reader = neo.io.BlackrockIO(filename=f'filepath')
    blk = reader.read_block()



# Converts Blackrock file and reads it into 'blk'
reader = neo.io.BlackrockIO(filename='/Users/maxtenenbaum/Desktop/Deku Analysis/120D1saline2002.ns6')
blk = reader.read_block()
print("Number of segments:", len(blk.segments))

# For each segment, print information about the contained data
for i, seg in enumerate(blk.segments):
    print("Segment", i)
    print("  Number of analog signals:", len(seg.analogsignals))
    print("  Number of spike trains:", len(seg.spiketrains))
    print("  Number of events:", len(seg.events))
if len(blk.segments) > 0 and len(blk.segments[0].analogsignals) > 0:
    analog_signal = blk.segments[0].analogsignals[0]
    print(analog_signal)
    # Optionally, convert to a NumPy array and print the shape
    analog_array = analog_signal.magnitude
    print('Array shape =',analog_array.shape)
    print('Max =',analog_array.max(), '// Min =', analog_array.min(), '// Mean =', analog_array.mean())