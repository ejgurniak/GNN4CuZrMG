'''
Given: a text file with the full Voronoi distribution for a structure
Output: a text file with a feature vector, each feature is the fraction of a Voronoi motif
Note: this is just the motif fraction
some motifs are Cu only, some are Zr only, however, this code just considers overall fraction
'''
# imports
import numpy as np
import os

num_atom = 686 # for 2.2 nm box, 686 atoms always
# change depending on the box size

# all_motifs is the list of motifs we plan to consider
# it is not all possible motifs in a sample
all_motifs = [[0, 0, 0, 0, 12, 0, 0], [0, 0, 0, 1, 10, 2, 0], \
    [0, 0, 0, 2, 8, 1, 0], [0, 0, 0, 2, 8, 2, 0], \
        [0, 0, 0, 2, 8, 3, 0], [0, 0, 0, 3, 6, 3, 0], \
            [0, 0, 0, 3, 6, 4, 0], [0, 0, 1, 0, 9, 3, 0], \
                [0, 0, 0, 0, 12, 3, 0], [0, 0, 0, 0, 12, 4, 0], \
                    [0, 0, 0, 1, 10, 4, 0], [0, 0, 0, 1, 10, 5, 0], \
                        [0, 0, 0, 1, 10, 6, 0], [0, 0, 0, 2, 8, 5, 0], \
                            [0, 0, 0, 2, 8, 6, 0], [0, 0, 0, 2, 8, 7, 0], \
                                ]
# open the file for reading

out_file = open('0.txt', 'w')
out_file.write("5\n") # check the dataset to find the true label
# loop through motifs and search in the Voronoi distribution
for i_motif in range(len(all_motifs)):
    input_file = open('Voronoi_distribution.txt', 'r')
    my_motif = all_motifs[i_motif]
    myline = input_file.readline()
    mylist = myline.split()
    # search for my motif
    found_my_motif = False
    while myline and found_my_motif == False:
        myline = input_file.readline()
        mylist = myline.split(",")
        alt_mylist = myline.split()
        matches = True
        if len(mylist) > 1:
            for i in range(1, 7, 1):
                if int(mylist[i]) != my_motif[i]:
                    matches = False
            if matches == True:
                found_my_motif = True
                break
    if len(mylist) == 1:
        num_my_motif = 0
    if found_my_motif:
        num_my_motif = int(alt_mylist[len(alt_mylist) - 3])
    else:
        num_my_motif = 0
    
    fraction_my_motif = num_my_motif / num_atom
    my_motif_str = f"<{my_motif[2]},{my_motif[3]},{my_motif[4]},{my_motif[5]}>"
    out_file.write(f"{my_motif_str}: {fraction_my_motif}\n")
    input_file.close()
out_file.close()

print("here")