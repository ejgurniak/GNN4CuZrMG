### Additional analysis that is relevant
- jensen-shannon_compare_1_100samples.py: Given: a file with the averaged g(r) (100 samples of the same box size, same class) and files with each g(r) this code will calculate the jensen-shannon distance between each individual sample g(r) and the ensemble g(r) and output the number to a results folder.
- temp_g_of_r.py: template for calculating g(r) from dump files. Replace "temp" with the seed/index of your sample.
- ensemble_100samples_g_of_r.py: Given: 100 individual g(r) results (all with the same number of bins, same bin size and cutoff) this code calculates the average (ensemble) g(r) for 100 samples of the same box size, same class.
