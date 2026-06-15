### Explanation of the dataset
- structure files are in POSCAR format
- the first line is the class label
- the rest of the file describes the structure of the sample
#### class labels
- "0": ~10<sup>15</sup> K/s
- "1": 10<sup>13</sup> K/s
- "2": 10<sup>12</sup> K/s
- "3": 10<sup>11</sup> K/s
- "4": 10<sup>10</sup> K/s
- "5": 10<sup>9</sup> K/s
#### how to use dataset to replicate our train-val split
- unzip all folders
- put data in one folder
- use the first 80 % for train, next 20 % for validation, set manually in custom_config.yaml. For example, in a dataset with 100 samples, put train_size: 80 and put val_size: 20 and put test_size: 0
#### additional pre-processing:
- write_POSCAR.py: reads a dump.quench.cfg file and writes a POSCAR file. The label will be replaced by the true label, and test.POSCAR will be replaced with index.POSCAR (0.POSCAR, 1.POSCAR, 2.POSCAR, etc.)
