
'''
calculates the jensen-shannon distance between each individual pair-wise g(r)
and the average g(r) over 100 samples of the box size, same class
Things you need:
1 - path to the normalized g(r), averaged over 100 samples
2 - path to each individual g(r), one for each sample
'''

# imports
from scipy.spatial import distance
import numpy as np

# important parameters
num_bins = 800 # 8 angstrom cutoff to be consistent
myclass = '9quench'
start_seed = 1026
num_samples = 100
end_seed = start_seed + num_samples - 1

# grab data for overall average g_of_r
avg_g_of_r = np.array([0.0]*num_bins)
in_file = open(f'../{myclass}_{start_seed}-{end_seed}_normalized_g_of_r.txt', 'r')
# TODO - replace the above file path with your file path
in_file.readline() # header
for i in range(num_bins):
    myline = in_file.readline()
    mylist = myline.split()
    # for j in range(3):
    avg_g_of_r[i] = float(mylist[1])
in_file.close()

# loop through the other samples and calculate jensen-shannon distance
for seed in range(start_seed, start_seed + num_samples, 1):
    # print(seed)
    single_g_of_r = np.array([0.0]*num_bins)
    in_file = open(f'../output/g_of_r_{myclass}{seed}.txt', 'r')
    # TODO - replace the above file path with your file path
    in_file.readline() # we don't need the first line
    second_line = in_file.readline()
    second_list = second_line.split()
    # print(second_list)
    if second_list[4] == 'g(r)':
        # print(second_line)
        for i in range(num_bins):
            myline = in_file.readline()
            mylist = myline.split()
            # for j in range(3):
            single_g_of_r[i] = float(mylist[1])
        
    else:
        print("ERROR: unexpected file format, expecting g(r) written by OVITO")

    in_file.close()
    mydistance = distance.jensenshannon(avg_g_of_r, single_g_of_r)
    if mydistance > 0.3:
        print(mydistance)

    # # Cu-Cu Cu-Zr Zr-Zr
    ## this time it's only one
    out_file = open(f'./results/{seed}.txt', 'w')
    out_file.write(str(mydistance))
    out_file.close()
