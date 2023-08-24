<h1 align="center">
    FOV Dropout Detection
</h1>

## Installation

To install the dropout detection tool simply run the following commands
```bash
$ git clone https://github.com/jacobquon/fov_dropouts.git
$ cd fov_dropouts
$ pip install .
```

Alternatively, if you don't want the workbook install directly from git using
```bash
$ pip install git+https://github.com/jacobquon/fov_dropouts.git
```

## How to use

Please refer to the demo for how to use the dropout detection package.

Should you like to run a basic pipeline across numerous barcodes a simple group of wrapper scripts is provided. Run the following command for more information:
```bash
$ python dropout_QC_slurm.py -h
```