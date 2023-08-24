from dropout_detection import DropoutResult
from pathlib import Path
from sys import argv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse

def main(barcodes, output_dir):
    results = []
    barcodes = barcodes.split(',')
    for barcode in barcodes:
        results.append(DropoutResult(output_dir / f"{barcode}/{barcode}_fovs.pkl"))

    fig, axs = plt.subplots(2,2, figsize=(20,10), layout='tight')

    fovs_dropped = []
    pct_fovs_dropped = []
    genes_dropped = []
    pct_genes_dropped = []
    
    for dr in results:
        fovs_dropped.append(dr.get_dropped_fov_counts())
        pct_fovs_dropped.append(dr.get_dropped_fov_counts() / max(dr.get_considered_fov_counts(), 1))
        genes_dropped.append(dr.get_dropped_gene_counts())
        pct_genes_dropped.append(dr.get_dropped_gene_counts() / max(dr.get_considered_gene_counts(), 1))

    p = axs[0, 0].bar(np.arange(len(barcodes)), fovs_dropped, width=0.8)
    p = axs[0, 1].bar(np.arange(len(barcodes)), pct_fovs_dropped, width=0.8)
    p = axs[1, 0].bar(np.arange(len(barcodes)), genes_dropped, width=0.8)
    p = axs[1, 1].bar(np.arange(len(barcodes)), pct_genes_dropped, width=0.8)
    
    for i in range(4):
        t = axs[i//2, i%2].set_xticks(ticks=np.arange(len(barcodes)), labels=barcodes, rotation = 45, ha="right", rotation_mode='anchor')

    axs[0, 0].set_title("Number of unique FOVs with dropout across experiments")
    axs[0, 0].set_ylabel("Unique FOVs with dropout")
    axs[0, 1].set_title("Percent of unique, considered FOVs with dropout across experiments")
    axs[0, 1].set_ylabel("Percent of unique, considered FOVs with dropout")
    axs[1, 0].set_title("Number of genes with dropout across experiments")
    axs[1, 0].set_ylabel("Genes with dropout")
    axs[1, 1].set_title("Percent of considered genes with dropout across experiments")
    axs[1, 1].set_ylabel("Percent of considered genes with dropout")
    
    fig.supxlabel("Experiments")
    fig.savefig(output_dir / 'dropout_barplot.png', format='png', bbox_inches='tight', dpi=400, facecolor="#FFFFFF", edgecolor="#FFFFFF", transparent=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--barcodes', type=str, help='comma seperated list of barcodes to run QC on')
    parser.add_argument('--output_dir', type=str, help='directory to output barplot into')
    
    config = parser.parse_args()
    main(config.barcodes, Path(config.output_dir))
