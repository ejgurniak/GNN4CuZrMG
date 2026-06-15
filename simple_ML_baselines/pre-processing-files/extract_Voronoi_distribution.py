'''
This code extracts the full Voronoi distribution from an atomic structure in a dump file
This code requires the OVITO in the conda environment
To change to extract multiple Voronoi distributions over multiple samples, 
change to a loop structure and provide paths to all of the samples

This code is modified from the original OVITO example found here:
https://www.ovito.org/manual/python/introduction/examples/batch_scripts/voronoi_indices.html
'''

# Import OVITO modules.
from ovito.io import *
from ovito.modifiers import *

# Import NumPy module.
import numpy

filepath = './dump.quench.cfg'
# import the dump file
pipeline = import_file(filepath)

def assign_particle_radii(frame, data):
    atom_types = data.particles_.particle_types_
    atom_types.type_by_id_(1).radius = 1.35 # Cu atomic radius assigned to atom type 1
    atom_types.type_by_id_(2).radius = 1.55 # Zr atomic radius assigned to type 2
pipeline.modifiers.append(assign_particle_radii)

# Set up the Voronoi analysis modifier
voro = VoronoiAnalysisModifier(
    compute_indices = True,
    use_radii = False,
)

pipeline.modifiers.append(voro)

# Let OVITO compute the results
data = pipeline.compute()

# Access computed Voronoi indices.
voro_indices = data.particles['Voronoi Index']

# This helper function takes a two-dimensional array and computes a frequency
# histogram of the data using some Numpy magic
# It returns two arrays (of equal length):
#   1. The listof unique data rows from the input array
#   2. The number of occurences of each unique row
# Both arrays are sorted in descending order such that the most frequent rows
# are listed first.
def row_histogram(a):
    ca = numpy.ascontiguousarray(a).view([('', a.dtype)] * a.shape[1])
    unique, indices, inverse = numpy.unique(ca, return_index=True, return_inverse=True)
    counts = numpy.bincount(inverse)
    sort_indices = numpy.argsort(counts)[::-1]
    return (a[indices[sort_indices]], counts[sort_indices])

# Compute frequency histogram.
unique_indices, counts = row_histogram(voro_indices)

num_unique_indices, _ = unique_indices.shape
out_file = open('./Voronoi_distribution.txt', 'w')
out_file.write(f"number of unique Voronoi indices = {num_unique_indices}\n")
for i in range(num_unique_indices):
    out_file.write("%s\t%i\t(%.1f %%)" % (tuple(unique_indices[i]),
                            counts[i],
                            100.0*float(counts[i])/len(voro_indices)))
    out_file.write("\n")

out_file.close()

print("here")