Code to complete an incomplete catalog with galaxies according to clustering.

How to use:
Download repository and run code with ./LaunchMyCompletionRuns.py 0. A different number given to this command produces a different random realization of a completed catalog.


A word on running time: two main parameters influence how fast the code is. 

"resol": the number of redshift shells, and therefore pixels, that are used in the code.
Resol = 50 produces a completed catalog in a little less than an hour. Resol=100 produces a completed catalog in several hours.

N_events: the number of gravitational wave events.
Each event takes around 7 seconds. I usually use 1000 GW events, for a total of ~2 hours.
