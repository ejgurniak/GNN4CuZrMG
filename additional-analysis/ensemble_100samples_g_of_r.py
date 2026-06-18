'''
Given: 100 samples with g(r) calculated
Output: ensemble g(r) averaged over 100 samples
Note: it is important that when you made the g(r), that you used the same number of bins for each g(r)
'''
# normalize the g(r) by looping over all of the indices

import numpy as np

# decide which 'class' or sample type over which we are doing a normalization
sample = '9quench'
start_id = 1026
end_id = 1125
num_samples = end_id - start_id + 1

overall_list = []

# loop through the files and add data to the lists
for i in range(start_id, end_id + 1, 1):
    filename = f"./output/g_of_r_{sample}{i}.txt"
    # TODO - change file path to your file path where you stored the g(r) output
    # from a previous code (or directly from OVITO user interface)
    file = open(filename, 'r')
    myline = file.readline()
    mylist = myline.split()
    overall_list.append(mylist)
    while myline:
        myline = file.readline()
        mylist = myline.split()
        overall_list.append(mylist)
    file.close()

# put only the relevant data into a numpy array
list_histogram_data_only = []
pointer = 0
while pointer < len(overall_list) - 1:
    if len(overall_list[pointer]) == 0:
        pointer += 1
    elif overall_list[pointer][0] == '#':
        pointer += 1
    else:
        list_histogram_data_only.append(overall_list[pointer])
        pointer += 1

num_bins = int((len(list_histogram_data_only)) / num_samples)

print(num_bins)
print(num_samples)

# initialize an un-normalized histogram
histogram = np.array([[0.0]*2]*num_bins)

# first num_bins lines in the list_histogram_data_only are the bin averages
# let's add those to the histogram array
for i in range(num_bins):
    histogram[i][0] = float(list_histogram_data_only[i][0])

for i in range(len(list_histogram_data_only)):
    mybin = i % num_bins
    histogram[mybin][1] += float(list_histogram_data_only[i][1])

# normalize the histogram
normalized_histogram = np.array([[0.0]*2]*num_bins)

for i in range(num_bins):
    normalized_histogram[i][0] = histogram[i][0]
    normalized_histogram[i][1] = histogram[i][1] / num_samples

filename = f"{sample}_{start_id}-{end_id}_normalized_g_of_r.txt"
# TODO - change to your preferred filename
print(filename)
out_file = open(filename, 'w')
first_line = f"{sample}_r {sample}_g_of_r"
out_file.write(first_line)
out_file.write("\n")
for i in range(len(normalized_histogram)):
    for j in range(2):
        out_file.write(str(normalized_histogram[i][j]))
        out_file.write(' ')
    out_file.write("\n")
out_file.close()
