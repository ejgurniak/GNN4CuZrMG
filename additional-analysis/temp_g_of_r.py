'''
Computes the g(r) of an atomic structure from a dump file
Note: This requires the OVITO in your conda environment
replace "temp" with the seed/index of your sample in pipeline and export_file
replace file paths to the path where you have your dump files, and where you want to save output
'''
from ovito.io import import_file, export_file
from ovito.modifiers import CoordinationAnalysisModifier

# Set up data pipeline:
pipeline = import_file("../dump_files/dump.quench.temp_9quench.cfg")
modifier = CoordinationAnalysisModifier(cutoff = 9.0, number_of_bins = 900)
pipeline.modifiers.append(modifier)
data = pipeline.compute()

# export the g(r) to a new file
export_file(pipeline, './output/g_of_r_9quenchtemp.txt', 'txt/table', key='coordination-rdf')
