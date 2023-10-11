import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import math
import tifffile as tiff
import subprocess
from pathlib import Path
from sys import argv
from time import time
import warnings
from matplotlib.colors import Normalize as color_normalize
import matplotlib.cm as cm
from numpy import ma

class DropoutResult:
    def __init__(self, fovs, transcripts=None, experiment_id=-1, merscope_out_dir='/allen/programs/celltypes/production/mfish', project="", region=0):
        """
        Initiate the dropout result
        
        Args:
            fovs (pd.DataFrame/str/Path): FOV table with stored results
            transcripts (pd.DataFrame) [default=None]: pandas dataframe of transcript information for the experiment
            experiment_id (int) [default=-1]: id for the experiment. Used to read in transcript information if transcripts argument is not provided
            merscope_out_dir (str/path) [default='/allen/programs/celltypes/production/mfish']: Merscopes output directory (contains all projects) for reading transcripts via experiment_id
   	    project (str) [default='']: Name of the project the experiment is in for reading transcripts via experiment_id. If unspecified, found automatically.
    	    region (int) [default=0]: region which holds the transcripts of interest for reading transcripts via experiment_id
        """
        if type(fovs) == type(pd.DataFrame()):
            self.fovs = fovs
        elif type(fovs) == type("") and fovs.endswith('pkl'):
            self.fovs = pd.read_pickle(fovs)
        elif type(fovs) == type(Path("")) and fovs.suffix == '.pkl':
            self.fovs = pd.read_pickle(fovs)
        elif type(fovs) == type("") or type(fovs) == type(Path("")):
            self.fovs = pd.read_table(fovs, sep="\t", header=0, index_col='fov')
        else:
             raise ValueError("Invalid type for FOV table")

        genes = []
        for col in self.fovs.columns:
            if col.startswith('delta'):
                genes.append(col.split('_')[-1])
        self.genes = genes

        if not transcripts is None:
            self.transcripts = transcripts
        else:
            if experiment_id != -1:
                self.transcripts = self.read_transcripts(merscope_out_dir, experiment_id, project=project, region=region)
            else:
                self.transcripts = None


    def get_lims_project_id(self, merscope_out_dir, experiment_id):
        """
        Finds the LIMS project for a given exerpiment id
        
        Args:
            merscope_out_dir (str/path): Merscopes output directory (contains all projects)
            experiment_id (int): Id for experiment of interest

        Raises:
            Exception: if LIMS project ID is not found for given barcode
        """
        for project in os.listdir(merscope_out_dir):
            if str(experiment_id) in os.listdir(os.path.join(merscope_out_dir, project)):
                return project
        
        raise Exception(f'Experiment {experiment_id} not found in Isilon\nExperiment may not be downloaded yet.\nDouble check barcode and try again.')
        
        
    def read_transcripts(self, merscope_out_dir, experiment_id, project="", region=0):
        """
        Reads and returns detected_transcripts.csv file for given experiment ID.
        
        Args:
            merscope_out_dir (str/path): Merscopes output directory (contains all projects)
            experiment_id (int): Id for experiment of interest
            project (str): Name of the project the experiment is in 
            region (int): region which holds the transcripts of interest
        """
        if project == "": project = self.get_lims_project_id(merscope_out_dir, experiment_id) 
        path = f'{merscope_out_dir}/{project}/{experiment_id}/region_{region}/detected_transcripts.csv'
        transcripts = pd.read_csv(path, header=0, usecols=["global_x", "global_y", "fov", "gene"])
        
        return transcripts[~transcripts.gene.str.startswith("Blank")]

    
    def get_dropout_count(self):
        """
        Get the total number of dropped FOVs.
        """
        return np.count_nonzero(self.fovs.filter(regex='dropout'))
    
    def get_dropped_genes(self, fov=-1, dic=False):
        """
        Get a list of the dropped genes. If an FOV is specified, gets the list of dropped genes for specified FOV. If dic=True, creates a dictionary of FOVs and dropped genes
        
        Args:
            fov (int) [default=-1]: If specified will return the dropped genes for the specified FOV
            dic (bool) [default=False]: If True, will return a dictionary of FOVs and dropped genes
        """
        if fov != -1:
            return list(self.fovs.filter(regex='dropout').columns[np.where(self.fovs.filter(regex='dropout').loc[fov])].str.replace('dropout_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = np.zeros((len(self.get_considered_fovs()),len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='dropout_'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: list(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return list(self.fovs.filter(regex='dropout').columns[np.any(self.fovs.filter(regex='dropout'), axis=0)].str.replace('dropout_', ''))

            
    def get_dropped_gene_counts(self, fov=-1, dic=False):
        """
        Get the number of dropped genes. If an FOV is specified, gets the number of dropped genes for specified FOV. If dic=True, creates a dictionary of FOVs and dropped gene counts
        
        Args:
            fov (int) [default=-1]: If specified will return the number of dropped genes for the specified FOV
            dic (bool) [default=False]: If True, will return a dictionary of FOVs and dropped gene counts
        """
        if fov != -1:
            return len(self.fovs.filter(regex='dropout').columns[np.where(self.fovs.filter(regex='dropout').loc[fov])].str.replace('dropout_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = np.zeros((len(self.get_considered_fovs()),len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='dropout_'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: len(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return len(self.fovs.filter(regex='dropout').columns[np.any(self.fovs.filter(regex='dropout'), axis=0)].str.replace('dropout_', ''))

    def get_dropped_fovs(self, gene='', dic=False):
        """
        Get a list of dropped FOVs. If a gene is specified, gets a list of dropped FOVs for specified gene. If dic=True return a dictionary of genes and dropped FOVs
        
        Args:
            gene (str) [default='']: If specified will return the dropped FOVs for the specified gene
            dic (bool) [default=False]: If True, will return a dictionary of genes and their dropped FOVs
        """
        if gene != '':
            return list(self.fovs[self.fovs[f'dropout_{gene}']].index)
        elif dic:
            fovs = np.array(self.fovs.index)
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = -np.ones((len(genes), len(fovs)), dtype=np.int64)
            fov_idx, gene_idx = np.where(self.fovs.filter(regex='dropout_'))
            arr[gene_idx, fov_idx] = fovs[fov_idx]
            return {gene: list(arr[i][arr[i] != -1]) for i, gene in enumerate(genes)}
        else:
            return list(self.fovs[self.fovs.filter(regex='dropout').sum(axis=1) > 0].index)

    def get_dropped_fov_counts(self, gene='', dic=False):
        """
        Get the number of unique dropped FOVs. If a gene is specified, gets the number of dropped FOVs for specified gene. If dic=True return a dictionary of genes and dropped FOV counts
        
        Args:
            gene (str) [default='']: If specified will return the dropped FOV count for the specified gene
            dic (bool) [default=False]: If True, will return a dictionary of genes and their dropped FOV counts
        """
        if gene != '':
            return len(self.fovs[self.fovs[f'dropout_{gene}']].index)
        elif dic:
            fovs = np.array(self.fovs.index)
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = -np.ones((len(genes), len(fovs)), dtype=np.int64)
            fov_idx, gene_idx = np.where(self.fovs.filter(regex='dropout_'))
            arr[gene_idx, fov_idx] = fovs[fov_idx]
            return {gene: len(arr[i][arr[i] != -1]) for i, gene in enumerate(genes)}
        else:
            return len(self.fovs[self.fovs.filter(regex='dropout').sum(axis=1) > 0].index)

    def get_considered_genes(self, fov=-1, dic=False):
        """
        Get a list of all genes with at least 1 FOV considered for dropout. An FOV is considered for dropout only if its 4 cardinal neighbors average at least 100 transcripts. 
        If an FOV is specified, gets list for the specified FOV
        If dic is set to True, returns a dictionary of considered genes for each considered fov
        
        Args:
            fov (int) [default=-1]: If specified will return the list of considered genes for the specified FOV
            dic (bool) [default=False]: If set to True, returns a dictionary of considered fovs and considered genes 
        """
        if fov != -1:
            return list(self.fovs.filter(regex='transcript_threshold').columns[np.where(self.fovs.filter(regex='transcript_threshold').loc[fov])[0]].str.replace('transcript_threshold_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='transcript_threshold').columns.str.replace('transcript_threshold_', ''))
            arr = np.zeros((len(self.get_considered_fovs()),len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='transcript_threshold'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: list(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return list(self.fovs.filter(regex='transcript_threshold').columns[np.sum(self.fovs.filter(regex='transcript_threshold'), axis=0) >= 1].str.replace('transcript_threshold_', ''))

    def get_considered_gene_counts(self, fov=-1, dic=False):
        """
        Get the number of genes with at least 1 FOV considered for dropout. An FOV is considered for dropout only if its 4 cardinal neighbors average at least 100 transcripts.
        If an FOV is specified, get the number of genes for which the FOV was considered
        If dic is set to True, returns a dictionary of the number of considered genes for each considered fov
        
        Args:
            fov (int) [default=-1]: If specified will return the number considered genes for the specified FOV
            dic (bool) [default=False]: If set to True, returns a dictionary of considered fovs and number of considered genes 
        """
        if fov != -1:
            return len(self.fovs.filter(regex='transcript_threshold').columns[np.where(self.fovs.filter(regex='transcript_threshold').loc[fov])[0]].str.replace('transcript_threshold_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='transcript_threshold').columns.str.replace('transcript_threshold_', ''))
            arr = np.zeros((len(self.get_considered_fovs()),len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='transcript_threshold'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: len(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return len(self.fovs.filter(regex='transcript_threshold').columns[np.sum(self.fovs.filter(regex='transcript_threshold'), axis=0) >= 1].str.replace('transcript_threshold_', ''))
            
    def get_considered_fovs(self):
        """
        Get a list of all on-tissue FOVs
        """
        return list(self.fovs[self.fovs['on_tissue']].index)

    def get_considered_fov_counts(self):
        """
        Get a the number of on-tissue FOVs
        """
        return len(self.fovs[self.fovs['on_tissue']].index)
    
    def get_false_positive_fovs(self, gene='', dic=False):
        """
        Get a list of all FOVs which were not considered dropped due to False Positive Correction. If a gene is specified, return the false positive FOVs for that gene. If dic=True, return a dictionary of false positive FOVs for each gene
        
        Args:
            gene (str) [default='']: If specififed, return false positive FOVs for that gene
            dic (bool) [default=False]: If True, return a dictionary of genes and their false positive FOVs
        """
        if gene != '':
            return list(self.fovs.loc[self.fovs['false_positive'].str.contains(gene), 'false_positive'].index)
        elif dic:
            return {gene: self.get_false_positive_fovs(gene=gene) for gene in self.genes}
        else:
            return list(self.fovs.iloc[np.where(self.fovs['false_positive'] != '')].index)

    def get_false_positive_fov_counts(self, gene='', dic=False):
        """
        Get a the number of FOVs which were not considered dropped due to False Positive Correction. If a gene is specified, return the number of false positive FOVs for that gene. If dic=True, return a dictionary of false positive FOV counts for each gene
        
        Args:
            gene (str) [default='']: If specififed, return false positive FOVs for that gene
            dic (bool) [default=False]: If True, return a dictionary of genes and their false positive FOVs
        """
        if gene != '':
            return len(self.fovs.loc[self.fovs['false_positive'].str.contains(gene), 'false_positive'].index)
        elif dic:
            return {gene: self.get_false_positive_fovs(gene=gene) for gene in self.genes}
        else:
            return len(self.fovs.iloc[np.where(self.fovs['false_positive'] != '')].index)

    def get_false_positive_genes(self, fov=-1, dic=False):
        """
        Get a list of all genes which had an FOV which was determined not dropped due to False Positive Correction. If an FOV is specified, get a list of all false positive genes for that FOV. If dic=True, return a dictionary of fovs and their false positive genes.
        
        Args:
            fov (int) [default=-1]: If specified, return false positive genes for a specific FOV
            dic (bool) [default=False]: If specified, return dictionary of FOVs and their false positive genes
        """
        if fov != -1:
            fp = self.fovs.loc[fov, 'false_positive']
            if len(fp) == 0:
                return []
            else:
                return fp.split(';')
        elif dic:
            return dict(self.fovs.loc[self.get_considered_fovs(), 'false_positive'].apply(lambda x: x.split(';') if len(x) > 0 else []))
        else:
            fp_genes = []
            for fov in self.get_false_positive_fovs():
                # Union of two sets to get unique genes
                fp_genes = list(set(fp_genes) | set(self.fovs.loc[fov, 'false_positive'].split(';')))
            return fp_genes

    def get_false_positive_gene_counts(self, fov=-1, dic=False):
        """
        Get the number genes which had an FOV which was determined not dropped due to False Positive Correction. If an FOV is specified, get the number of false positive genes for that FOV. If dic=True, return a dictionary of fovs and their false positive gene counts
        
        Args:
            fov (int) [default=-1]: If specified, return the number of false positive genes for a specific FOV
            dic (bool) [default=False]: If specified, return dictionary of FOVs and their false positive gene counts
        """
        if fov != -1:
            fp = self.fovs.loc[fov, 'false_positive']
            if len(fp) == 0:
                return 0
            else:
                return len(fp.split(';'))
        elif dic:
            return dict(self.fovs.loc[self.get_considered_fovs(), 'false_positive'].apply(lambda x: len(x.split(';')) if len(x) > 0 else 0))
        else:
            fp_genes = []
            for fov in self.get_false_positive_fovs():
                # Union of two sets to get unique genes
                fp_genes = list(set(fp_genes) | set(self.fovs.loc[fov, 'false_positive'].split(';')))
            return len(fp_genes)
            
    
    def dropout_summary(self, return_summary=False):
        """
        Prints a summary of the dropout for the experiment
        
        Args:
            return_summary (bool) [default=False]: if True, returns the summary as a string
        """
        summary = f"{self.get_dropped_fov_counts()} unique FOVs were dropped out of {self.get_considered_fov_counts()} considered FOVs ({100 * self.get_dropped_fov_counts() / max(self.get_considered_fov_counts(), 1):.2f}%)\n" +\
               f"FOVs with dropout dropped out in {self.get_dropout_count() / max(self.get_dropped_fov_counts(), 1):.2f} genes on average out of possible total of {self.get_considered_gene_counts()} ({100 * self.get_dropout_count() / max(self.get_considered_gene_counts() * self.get_dropped_fov_counts(), 1):.2f}%)\n" +\
               f"{self.get_false_positive_fov_counts()} unique FOVs were not considered dropped due to false positive correction ({100 * self.get_false_positive_fov_counts() / max(self.get_false_positive_fov_counts() + self.get_dropped_fov_counts(), 1):.2f}% of the FOVs initially considered dropped)\n" +\
               f"{self.get_dropped_gene_counts()} genes were affected by dropout out of {self.get_considered_gene_counts()} possible ({100 * self.get_dropped_gene_counts() / max(self.get_considered_gene_counts(), 1):.2f}%)\n" +\
               f"Genes with dropout averaged {self.get_dropout_count() / max(self.get_dropped_gene_counts(), 1):.2f} dropped FOVs out of {self.get_considered_fov_counts()} possible FOVs ({100 * self.get_dropout_count() / max(self.get_dropped_gene_counts() * self.get_considered_fov_counts(), 1):.2f}%)"
        print(summary)
        if return_summary:
            return summary
        
    def draw_dropout(self, gene, out_file=''):
        """
        Draws the detected dropout for a gene. Transcripts are drawn for the gene in the left plot. The right plot also contains the transcripts but also highlights dropped FOVs in blue
        
        Args:
            gene (str): The gene to draw the detected dropout fov
            out_file (str) [default='']: If specified, the function will output the plot to the specified path and close
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        plot_min = min((np.min(self.transcripts['global_x']), np.min(self.transcripts['global_y'])))
        plot_max = max((np.max(self.transcripts['global_x']), np.max(self.transcripts['global_y'])))
        plt.setp(ax, xlim=(plot_min, plot_max), ylim=(plot_max, plot_min))
    
        ax[0].scatter(self.transcripts[self.transcripts['gene'] == gene]['global_x'], self.transcripts[self.transcripts['gene'] == gene]['global_y'], s=0.001, c='k', marker='.')
        ax[1].scatter(self.transcripts[self.transcripts['gene'] == gene]['global_x'], self.transcripts[self.transcripts['gene'] == gene]['global_y'], s=0.001, c='k', marker='.')
    
        ax[0].set_title(gene)
        ax[1].set_title(gene)
        
        grid_sq_size = max((np.max(self.fovs['width']), np.max(self.fovs['height'])))
        
        for fov in self.get_dropped_fovs(gene):
            ax[1].add_patch(Rectangle((self.fovs.loc[fov, "x_min"], self.fovs.loc[fov, "y_min"]),
                   grid_sq_size,
                   grid_sq_size,
                   facecolor = 'blue',
                   fill=True,
                   lw=0,
                   alpha=0.5))
        
        if out_file != '':
            fig.savefig(out_file, format='png', bbox_inches='tight', dpi=400, facecolor="#FFFFFF", edgecolor="#FFFFFF", transparent=False)
            plt.close()

    def draw_total_dropout_map(self, out_file=''):
        """
        Draws heatmaps of number of genes considered for dropout per FOV and number of genes dropped out per FOV

        Args:
            out_file (str) [default='']: If specified, the function will output the plot to the specified path and close
        """
        
        fig, ax = plt.subplots(1, 5, figsize=(32.5, 5))
        
        # Create dictionaries with keys as FOVs and values as considered/dropped genes
        genes_considered_per_fov = self.get_considered_gene_counts(dic=True)
        genes_dropped_per_fov = self.get_dropped_gene_counts(dic=True)
        fp_genes_per_fov = self.get_false_positive_gene_counts(dic=True)
        drop_div_per_fov = {fov: genes_dropped_per_fov[fov] / genes_considered_per_fov[fov] for fov in self.get_considered_fovs()}
        fp_div_per_fov = {fov: fp_genes_per_fov[fov] / genes_considered_per_fov[fov] for fov in self.get_considered_fovs()}
        
        grid_sq_size = max((np.max(self.fovs['width']), np.max(self.fovs['height'])))
        
        plot_min = min((np.min(self.transcripts['global_x']), np.min(self.transcripts['global_y'])))
        plot_max = max((np.max(self.transcripts['global_x']), np.max(self.transcripts['global_y'])))
        plt.setp(ax, xlim=(plot_min, plot_max), ylim=(plot_max, plot_min))
        
        ax[0].set_title('Genes Considered by FOV')
        ax[1].set_title('Genes Dropped by FOV')
        ax[2].set_title('Genes Dropped by FOV / Genes Considered by FOV')
        ax[3].set_title('False Positive Gene Counts by FOV')
        ax[4].set_title('False Positive Gene Counts by FOV / Genes Considered by FOV')
        
        # Genes considered by FOV
        norm = color_normalize(vmin=0, vmax=np.max(list(genes_considered_per_fov.values())))
        for fov, genes_considered in genes_considered_per_fov.items():
            ax[0].add_patch(Rectangle((self.fovs.loc[fov, "x_min"], self.fovs.loc[fov, "y_min"]),
                   grid_sq_size,
                   grid_sq_size,
                   facecolor = cm.viridis(norm(genes_considered)),
                   fill=True,
                   lw=0))
        self.determine_ticks(fig, ax[0], norm, np.max(list(genes_considered_per_fov.values())))
        
        # Genes dropped by FOV
        norm = color_normalize(vmin=0, vmax=max(np.max(list(genes_dropped_per_fov.values())), 1))        
        for fov, genes_dropped in genes_dropped_per_fov.items():
            ax[1].add_patch(Rectangle((self.fovs.loc[fov, "x_min"], self.fovs.loc[fov, "y_min"]),
                   grid_sq_size,
                   grid_sq_size,
                   facecolor = cm.viridis(norm(genes_dropped)),
                   fill=True,
                   lw=0))
        self.determine_ticks(fig, ax[1], norm, max(np.max(list(genes_dropped_per_fov.values())), 1))
        
        # Genes dropped by FOV / Genes considered by FOV
        norm = color_normalize(vmin=0, vmax=max(np.max(list(drop_div_per_fov.values())), .01))        
        for fov, drop_div in drop_div_per_fov.items():
            ax[2].add_patch(Rectangle((self.fovs.loc[fov, "x_min"], self.fovs.loc[fov, "y_min"]),
                   grid_sq_size,
                   grid_sq_size,
                   facecolor = cm.viridis(norm(drop_div)),
                   fill=True,
                   lw=0))
        self.determine_ticks(fig, ax[2], norm, max(np.max(list(drop_div_per_fov.values())), .01))
        
        # False positive genes by FOV
        norm = color_normalize(vmin=0, vmax=max(np.max(list(fp_genes_per_fov.values())), 1))
        for fov, fp_genes in fp_genes_per_fov.items():
            ax[3].add_patch(Rectangle((self.fovs.loc[fov, "x_min"], self.fovs.loc[fov, "y_min"]),
                   grid_sq_size,
                   grid_sq_size,
                   facecolor = cm.viridis(norm(fp_genes)),
                   fill=True,
                   lw=0))
        self.determine_ticks(fig, ax[3], norm, max(np.max(list(fp_genes_per_fov.values())), 1))
        
        # False positive genes by FOV / Genes considered by FOV
        norm = color_normalize(vmin=0, vmax=max(np.max(list(fp_div_per_fov.values())), .01))    
        for fov, fp_div in fp_div_per_fov.items():
            ax[4].add_patch(Rectangle((self.fovs.loc[fov, "x_min"], self.fovs.loc[fov, "y_min"]),
                   grid_sq_size,
                   grid_sq_size,
                   facecolor = cm.viridis(norm(fp_div)),
                   fill=True,
                   lw=0))
        self.determine_ticks(fig, ax[4], norm, max(np.max(list(fp_div_per_fov.values())), .01))
        
        if out_file != '':
            fig.savefig(out_file, format='png', bbox_inches='tight', dpi=400, facecolor="#FFFFFF", edgecolor="#FFFFFF", transparent=False)
            plt.close()
        
    def determine_ticks(self, fig, ax, norm, max_tick):
        """
        Helper for determining how many ticks to add to the colorbars for the total dropout maps
        
        Args:
            fig (matplotlib figure): figure to draw the colorbar on
            ax (matplotlib axis): axis to draw the colorbar on
            norm (matplotlib.colors.Normalize): normalization used for the colormap
            max_tick (int): The highest tick mark to be plotted on the colorbar
            log
        """
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax)
        if max_tick <= 0.1:
            step_size = 0.01
        elif max_tick <= 1:
            step_size = 0.1
        elif max_tick <= 10:
            step_size = 1
        elif max_tick < 100:
            step_size = 10
        elif max_tick < 200:
            step_size = 20
        else:
            step_size = 50
        ticks = np.concatenate((np.arange(0, max_tick, step_size), np.array([max_tick])))
        if len(ticks) == 1:
            cb.set_ticks(np.array([0, 1]))
            cb.set_ticklabels(np.array([0, 1]))
        else:
            cb.set_ticks(ticks)
            cb.set_ticklabels(np.around(ticks, 3))

    def draw_top_mid_bot(self, output_dir):
        """
        Draw (by transcript count) the top 20 genes, bottom 10 gene, and a subset of 10 genes in the center starting from the last gene to average 100 transcripts per FOV
        
        Args:
            output_dir (str/Path): the directory to output the drawings into
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError as e:
            pass
        
        # Order the genes by transcript count
        gene_counts = self.transcripts.groupby('gene').size()
        gene_counts = gene_counts.sort_values(ascending=False)
        
        # Draw the top 20 highest transcript count genes
        for gene in gene_counts.iloc[:20].index:
            self.draw_dropout(gene, out_file=os.path.join(output_dir, f"{gene}.png"))
            
        # Draw the 10 lowest transcript count genes
        for gene in gene_counts.iloc[-10:].index:
            self.draw_dropout(gene, out_file=os.path.join(output_dir, f"{gene}.png"))
        
        # Draw a subset of 10 genes in the center starting from the last gene to average 100 transcripts per FOV
        idx = np.max(np.where(gene_counts > len(self.get_considered_fovs()) * 100)[0])
        for gene in gene_counts.iloc[idx:idx+10].index:
            self.draw_dropout(gene, out_file=os.path.join(output_dir, f"{gene}.png"))
            
    def draw_dropped_genes(self, output_dir, max_genes=-1):
        """
        Draw all the dropped genes
        
        Args:
            output_dir (str/Path): the directory to output the drawings into
            max_genes (int) [default=-1]: if specified, only draw the top N dropped genes (ordered by transcript count)
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError as e:
            pass
        
        # Order genes by transcript count
        gene_counts = self.transcripts.groupby('gene').size()
        gene_counts = gene_counts.sort_values(ascending=False)

        genes_drawn = 0
        # Loop over all genes
        for gene in self.genes:
            # If we have a max number and we have drawn more than that max, stop drawing
            if max_genes != -1 and genes_drawn >= max_genes:
                break
            # If the gene has been dropped, draw it
            if np.count_nonzero(self.fovs[f'dropout_{gene}']) > 0:
                self.draw_dropout(gene, out_file=os.path.join(output_dir, f'{gene}.png'))
                genes_drawn += 1

class TranscriptImage(DropoutResult):
    # Init
    def __init__(self, experiment_id, merscope_out_dir='/allen/programs/celltypes/production/mfish', project="", region=0):
        """
        Reads in a merscope image into a transcript dataframe
        
        Args:
            experiment_id (str/int): experiment barcode to search MERSCOPE output folders for
            merscope_out_dir (str/Path): directory where merscope results are stored
            project (str) [default=""]: name of the project the experiment is in
            region (int) [default=0]: region of the experiment to run on
        """
        self.experiment_id = experiment_id
        # Read in data
        print("Reading transcripts", flush=True); t = time()
        self.transcripts = self.read_transcripts(merscope_out_dir, experiment_id, project=project, region=region)
        print("    Time:", time() - t, flush=True)

    def run_dropout_pipeline(self, mask_out_dir, codebook_path, threshold=0.15):
        """
        Runs all the steps necessary for dropout detection
        
        Args:
            mask_out_dir (str/path): the output directory for the tissue mask information (used to find on-tissue FOVs)
            codebook_path (str/path): The path to the codebook used in the merscope experiment
            threshold (float) [default=0.1]: The transcript delta threshold b/w an FOV and its neighbors at which the FOV is considered dropped
        """
        # Create FOV table
        print("Creating FOV table", flush=True); t = time()
        self.find_fovs()
        print("    Time:", time() - t, flush=True)

        # Find on-tissue FOVs
        print("Finding on-tissue FOVs", flush=True); t = time()
        self.find_on_tissue_fovs(mask_out_dir)
        print("    Time:", time() - t, flush=True)

        # Find the neighbors for each FOV
        print("Finding neighbors", flush=True); t = time()
        self.find_neighbors()
        print("    Time:", time() - t, flush=True)

        # Find dropout
        print("Detecting dropout", flush=True); t = time()
        self.detect_dropouts(threshold)
        print("    Time:", time() - t, flush=True)
        
        # Read in codebook
        print("Detecting false positives", flush=True); t = time()
        self.read_codebook(codebook_path)

        # Run False positive detection
        self.detect_false_positives()
        print("    Time:", time() - t, flush=True)

        super().__init__(self.fovs, transcripts=self.transcripts)
    
    def find_fovs(self):
        """
        Creates a dataframe to store information about FOVs and reads in preliminary information for each
        Also creates a dataframe which stores per-fov transcript counts for every gene

        Sets:
            self.fovs (dataframe): dataframe of all FOVs and their metadata
            self.transcripts_by_gene (dataframe): dataframe of all FOVs and how many transcripts there are for each gene
        """

        gene_fovs = self.transcripts[["global_x", "global_y", "fov"]].groupby('fov').min()
        gene_fovs.rename(columns={"global_x": "x_min", "global_y": "y_min"}, inplace=True)
        gene_fovs[["x_max", "y_max"]] = self.transcripts[["global_x", "global_y", "fov"]].groupby('fov').max()
        
        gene_fovs["width"] = gene_fovs["x_max"] - gene_fovs["x_min"]
        gene_fovs["height"] = gene_fovs["y_max"] - gene_fovs["y_min"]
        
        gene_fovs["center_x"] = (gene_fovs["x_max"] - gene_fovs["x_min"]) / 2 + gene_fovs["x_min"]
        gene_fovs["center_y"] = (gene_fovs["y_max"] - gene_fovs["y_min"]) / 2 + gene_fovs["y_min"]

        self.fovs = gene_fovs

        self.transcripts_by_gene = self.transcripts.groupby(['fov'])['gene'].value_counts().unstack(fill_value=0)

    def rounddown_10(self, x):
        '''
        Helper function that rounds down to the nearest 10
        '''
        return int(math.floor(x / 10.0)) * 10
    
    def ilastik_tissue_mask(self, mask_out_dir):
        '''
        This function utilizes the Ilastik program to generate a tissue mask for on-tissue transcript calculations. An image
        of all filtered transcripts is generated, fed into the ilastik pixel classification workflow to generate a probability
        map, which is then fed into the ilastik object classification workflow to assign features to objects, in this case, 
        tissue and non-tissue. 
        
        Args:
            mask_out_dir (str/Path): the directory in which to store all the tissue mask calculations
        Returns: 
            None, but images are generated and saved in specified folders
        '''
        try:
            mask_out_dir = Path(mask_out_dir)
            os.makedirs(mask_out_dir / 'images/')
            os.mkdir(mask_out_dir / 'masks/')
            os.mkdir(mask_out_dir / 'logs/')
            os.mkdir(mask_out_dir / 'probability_maps/')
        except FileExistsError as e:
            pass
        
        ilastik_folder = '/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/merfish_qc/tissue_mask'
            
        # Ilastik workflow
        ilastik_location = Path(f"{ilastik_folder}/ilastik_program/ilastik-1.4.0-Linux/run_ilastik.sh")
        pixel_project_location = Path(f"{ilastik_folder}/models/TissueMaskPixelClassification_v1.0.ilp")
        object_project_location = Path(f"{ilastik_folder}/models/TissueMaskObjects_v1.0.ilp")
        
        # convert filtered transcripts to an image via 2D histogram
        # The image goes from 0 to the the max x/y rounded to the nearest 10
        # Each pixel in the image represent 10 um, thus bin coordinates are original coords / 10
        # h = plt.hist2d(self.transcripts["global_x"], self.transcripts["global_y"], bins=(int(np.ceil(np.max(self.transcripts["global_x"])/10)), int(np.ceil(np.max(self.transcripts["global_y"])/10))))
        h, self.mask_x_bins, self.mask_y_bins, _ = plt.hist2d(self.transcripts["global_x"],
                                                              self.transcripts["global_y"],
                                                              bins=[np.arange(min(self.rounddown_10(np.min(self.transcripts["global_x"])), 0), int(np.ceil(np.max(self.transcripts["global_x"]))) + 10, 10),
                                                                    np.arange(min(self.rounddown_10(np.min(self.transcripts["global_y"])), 0), int(np.ceil(np.max(self.transcripts["global_y"]))) + 10, 10)])
        image_path = mask_out_dir / f'images/{self.experiment_id}.tif'
        tiff.imwrite(image_path, h)
        plt.close();

        out_log = mask_out_dir / f'logs/{self.experiment_id}.out'
        err_log = mask_out_dir / f'logs/{self.experiment_id}.err'
    
        out_log_file = open(out_log, 'w+')
        err_log_file = open(err_log, 'w+')
        
        # Pixel classification workflow: generating probability maps using raw images
        probability_map_path = mask_out_dir / f"probability_maps/{self.experiment_id}_probability_map.tiff"
        subprocess.run(
            [
                ilastik_location,
                "--headless",
                "--project",
                pixel_project_location,
                "--export_source",
                "Probabilities",
                "--raw_data",
                image_path,
                "--output_format=tiff",
                "--output_filename_format",
                probability_map_path,
            ],
            stdout=out_log_file,
            stderr=err_log_file
        )
        
        # Object classification workflow: generating object predictions using raw images and probability maps
        tissue_mask_path = mask_out_dir / f"masks/{self.experiment_id}_tissue_mask.tiff"
        subprocess.run(
            [
                ilastik_location,
                "--headless",
                "--project",
                object_project_location,
                "--export_source",
                "Object Predictions",
                "--prediction_maps",
                probability_map_path,
                "--raw_data",
                image_path,
                "--output_format=tiff",
                "--output_filename_format",
                tissue_mask_path,
            ],
            stdout=out_log_file,
            stderr=err_log_file
        )
        out_log_file.close()
        err_log_file.close()
        return None
    
    def find_on_tissue_fovs(self, mask_out_dir):
        '''
        Uses output from Ilastik to determine on and off-tissue FOVs, and then removes off-tissue FOVs from the FOV table
        An FOV is considered on-tissue if at least 50% of its area is on-tissue
        
        Args:
            mask_out_dir (str/Path): the directory where all the tissue mask calculations were stored
        
        Sets: 
            self.fovs (dataframe): Adds a column to track whether an FOV is on-tissue or not
        '''
        mask_out_dir = Path(mask_out_dir)
        self.ilastik_tissue_mask(mask_out_dir)
    
        # Coords map from the mask to 10x in the transcripts/gene_fovs coordinates
        # e.g. mask[472, 501] --> transcripts[4720:4730, 5010:5020]
        mask = tiff.imread(mask_out_dir / f"masks/{self.experiment_id}_tissue_mask.tiff")
    
        on_tissue = []
        for fov in self.fovs.index:
            # Find the mask coordinates of the FOV of interest
            mask_min_x = np.digitize(self.fovs.loc[fov, "x_min"], self.mask_x_bins) - 1
            mask_max_x = np.digitize(self.fovs.loc[fov, "x_max"], self.mask_x_bins)
            mask_min_y = np.digitize(self.fovs.loc[fov, "y_min"], self.mask_y_bins) - 1
            mask_max_y = np.digitize(self.fovs.loc[fov, "y_max"], self.mask_y_bins)
            
            # If 50% of the FOV is on-tissue, we consider it on-tissue
            if np.sum(mask[mask_min_x:mask_max_x, mask_min_y:mask_max_y]) >= 0.5 * (mask_max_x - mask_min_x) * (mask_max_y - mask_min_y) and self.fovs.loc[fov, 'height'] > 0 and self.fovs.loc[fov, 'width'] > 0:
                on_tissue.append(True)
            else:
                on_tissue.append(False)
            
        self.fovs['on_tissue'] = on_tissue

    def find_neighbors(self):
        """
        Find the neighbors for each of the on-tissue FOVs using the grid coordinates

        Sets:
            self.fovs (dataframe): modifies self.fovs to contain neighbor information
        """
        max_width = np.max(self.fovs["width"])
        max_height = np.max(self.fovs["height"])
        self.grid_sq_size = max((max_width, max_height))
        
        centers_arr = np.array(self.fovs[["center_x", "center_y"]])
        
        neighbors = [[] for i in range(len(self.fovs))]

        for i in range(len(self.fovs)):
            fov = self.fovs.index[i]
            # Find the euclidean distance between the grid coordinates of each FOV and all other FOVs
            fov_center = np.broadcast_to(centers_arr[i], (len(centers_arr), 2))
            euclidean_distances = np.argsort(np.linalg.norm(fov_center - centers_arr, axis=1))
            
            # Neighbor above
            above_fovs = np.unique(np.where((centers_arr[:,1] > self.fovs.loc[fov, "y_max"]) & (abs(centers_arr[:,0] - centers_arr[i,0]) <= self.grid_sq_size / 2) & (abs(centers_arr[:,1] - centers_arr[i,1]) <= self.grid_sq_size * 1.5))[0])
            if len(above_fovs) > 0:
                neighbors[i].append(self.fovs.index[euclidean_distances[np.isin(euclidean_distances, above_fovs)][0]])
            
            # Neighbor below
            below_fovs = np.unique(np.where((centers_arr[:,1] < self.fovs.loc[fov, "y_min"]) & (abs(centers_arr[:,0] - centers_arr[i,0]) <= self.grid_sq_size / 2) & (abs(centers_arr[:,1] - centers_arr[i,1]) <= self.grid_sq_size * 1.5))[0])
            if len(below_fovs) > 0:
                neighbors[i].append(self.fovs.index[euclidean_distances[np.isin(euclidean_distances, below_fovs)][0]])
                
            # Neighbor right
            right_fovs = np.unique(np.where((centers_arr[:,0] > self.fovs.loc[fov, "x_max"]) & (abs(centers_arr[:,1] - centers_arr[i,1]) <= self.grid_sq_size / 2) & (abs(centers_arr[:,0] - centers_arr[i,0]) <= 400))[0])
            if len(right_fovs) > 0:
                neighbors[i].append(self.fovs.index[euclidean_distances[np.isin(euclidean_distances, right_fovs)][0]])
            
            # Neighbor left
            left_fovs = np.unique(np.where((centers_arr[:,0] < self.fovs.loc[fov, "x_min"]) & (abs(centers_arr[:,1] - centers_arr[i,1]) <= self.grid_sq_size / 2) & (abs(centers_arr[:,0] - centers_arr[i,0]) <= 400))[0])
            if len(left_fovs) > 0:
                neighbors[i].append(self.fovs.index[euclidean_distances[np.isin(euclidean_distances, left_fovs)][0]])
            
        self.fovs["neighbors"] = neighbors

    def detect_dropouts(self, threshold):
        """
        Looks at the cardinal neighbors for each FOV to detect dropout. 
        An FOV is considered dropped IF 
            - it is below the delta threshold for number of transcripts for ALL 4 neighbors
            - OR it is below threshold for 3 neighbors and the last is also a dropped FOV
            
        Args: 
            threshold (float) - the delta threshold for dropout detection
            
        Sets:
            self.fovs (dataframe): Adds columns for each gene/fov pair for the deltas, whether it dropped, and whether it's neighbors were above 100 avg transcripts
        """
        gene_counts = self.transcripts.groupby('gene').size()
        gene_counts = gene_counts.sort_values(ascending=False)
        # Find dropout for every gene
        for gene in gene_counts.index:
            # Count the number of transcripts for each FOV for all transcripts of the gene of interest
            gene_df = self.transcripts_by_gene[gene].to_frame()

            # Do a first pass on 3 sides, so that we can detect FOV dropout when 2 are adjacent
            fov_deltas = []
            fov_dropout = []
            for fov in self.fovs.index:
                if self.fovs.loc[fov, 'on_tissue']: # Only want to detect dropout on tissue
                    neighbors = self.fovs.loc[fov, "neighbors"]
    
                    # Calculate the ratio between the number of transcripts in the FOV and its neighbors (we take max with 1 so that we don't divide 0 on either side)
                    # zero on left is bad because it will always be below delta then and zero on right is bad because divide by zero error 
                    deltas = np.array([max(gene_df.loc[fov, gene], 1) / max(gene_df.loc[neighbor, gene], 1) for neighbor in neighbors])
    
                    fov_deltas.append(deltas)
                    # only dropout if less than threshold on at least 3 sides
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fov_dropout.append(True if np.count_nonzero(deltas < threshold) >= 3 else False)
                else:
                    fov_deltas.append(np.array([]))
                    fov_dropout.append(False)
                    
            gene_df[f"deltas_{gene}"] = fov_deltas
            gene_df[f"dropout_{gene}"] = fov_dropout

            transcript_threshold = [] # Want to store whether or not the FOV was even considered (>100 avg transcripts among neighbors)
            # Do a second pass of 4 sides minus those which are a adjacent other dropout
            for fov in self.fovs.index:
                if self.fovs.loc[fov, 'on_tissue']: # Only want to detect dropout if it's on tissue              
                    neighbors = self.fovs.loc[fov, "neighbors"]
    
                    deltas = gene_df.loc[fov, f"deltas_{gene}"]
    
                    # Calculate the average transcripts of the non-dropped neighbors
                    neighbor_avg_transcripts = []
                    for neighbor in neighbors:
                        if not gene_df.loc[neighbor, f"dropout_{gene}"]:
                            neighbor_avg_transcripts.append(gene_df.loc[neighbor, gene])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        above_transcript_threshold = np.mean(neighbor_avg_transcripts) >= 100
                        transcript_threshold.append(above_transcript_threshold)
                        
                    if not above_transcript_threshold: # if not >= 100 average neighboring transcripts cannot be considered dropped
                        gene_df.loc[fov, f"dropout_{gene}"] = False
                    elif np.count_nonzero(deltas < threshold) == 4: # If less than threshold on 4 sides, its dropped
                        gene_df.loc[fov, f"dropout_{gene}"] = True
                    elif np.count_nonzero(deltas < threshold) == 3 and np.any(gene_df.loc[neighbors, f"dropout_{gene}"]): # if less than threshold on 3 sides and one of those sides is a dropped FOV, then its dropped
                        gene_df.loc[fov, f"dropout_{gene}"] = True
                    else: # Otherwise, it is not a dropped FOV
                        gene_df.loc[fov, f"dropout_{gene}"] = False
                else:
                    fov_deltas.append(np.array([]))
                    fov_dropout.append(False)
                    transcript_threshold.append(False)

            gene_df[f"transcript_threshold_{gene}"] = transcript_threshold
            self.fovs = self.fovs.merge(gene_df[[f"transcript_threshold_{gene}", f"deltas_{gene}", f"dropout_{gene}"]], left_index=True, right_index=True, how='left', suffixes=('', '_remove'))
            # remove the duplicate columns
            self.fovs.drop([i for i in self.fovs.columns if 'remove' in i], axis=1, inplace=True)

    def read_codebook(self, codebook_path):
        """
        Reads in a merscope codebook
        
        Args:
            codebook_path (str/Path): path to the merscope codebook file to be used for false positive correction
        """
        codebook = pd.read_table(codebook_path, header=0, sep=',').drop(columns=['id', 'barcodeType'], errors='ignore').set_index('name')
        self.codebook = codebook[~codebook.index.str.startswith("Blank")]

        # Warn user of potential problem if codebook genes are not the same as transcripts.csv genes and remove these genes from the fovs dataframe
        missing_genes_bool = False
        missing_transcripts_genes = []
        missing_codebook_genes = []
        transcripts_genes = list(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
        codebook_genes = list(self.codebook.index)
        for transcripts_gene in transcripts_genes:
            if transcripts_gene not in codebook_genes:
                missing_transcripts_genes.append(transcripts_gene)
                missing_genes_bool = True
                self.fovs = self.fovs.drop(columns=list(self.fovs.filter(regex=transcripts_gene))) # Drop the missing gene from the fovs dataframe
        for codebook_gene in codebook_genes:
            if codebook_gene not in transcripts_genes:
                missing_codebook_genes.append(codebook_gene)
                missing_genes_bool = True
        if missing_genes_bool:
            warnings.warn(f'WARNING: codebook and transcripts.csv contain differing genes. These genes will be removed from dropout consideration. Codebook is missing {",".join(missing_codebook_genes)}. Transcripts.csv is missing {",".join(missing_transcripts_genes)}')
        
    def detect_false_positives(self):
        """
        Detects false positive FOV dropouts by evaluating the codebook rounds in which the genes dropped for the particular FOV
        
        Sets:
            self.fovs (dataframe): modifies the dropout_{gene} entries to false if it detected a false positive
            self.fovs (dataframe): Adds a column which contains empty strings if an FOV was not a false positive and a comma separated list of genes if it was a false positive
        """
        
        # Create a dataframe which stores information on imaging rounds (3 codebook bits to a round)
        bits = self.codebook.shape[1]
        round_df = self.codebook.groupby((np.arange(bits) // 3) + 1, axis=1).sum()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            self.fovs['false_positive'] = ["" for i in range(len(self.fovs))]

        # Loop over all dropped FOVs
        for fov in self.fovs[np.any(self.fovs.filter(regex='dropout'), axis=1)].index:
            dropped_genes = list(self.fovs.filter(regex='dropout').columns[np.where(self.fovs.filter(regex='dropout').loc[fov])].str.replace('dropout_', ''))
            
            # Find the distribution of imaging rounds for all genes that experienced dropout in the FOV
            round_freqs = np.array(round_df.loc[dropped_genes].astype(bool).sum(axis=0) / len(dropped_genes))

            # if one of the rounds has is present in 100% of the genes, we know it is truly dropped
            # Must have 5 or more genes to be considered since its quite frequent that 3 or 4 would have 100% in a round
            if np.max(round_freqs) == 1 and len(dropped_genes) >= 5: 
                continue
                
            # If there are two high-scoring rounds, we need to determine if this is random chance or due to the FOV being dropped in 2 rounds
            # We could consider if there are 3 high-scoring rounds, but this should be very unlikely to actually happen and will bring the distribution too close to uniform

            # The threshold is determined by the chance of one high-scoring round not having a bit in the other high-scoring round
            # We know there are 4 positive bits for each gene code, which is why there are 3 divisions
            round_freq_threshold = 0.5 + 0.5 * (1 - (((bits - 4) / (bits - 1)) * ((bits - 5) / (bits - 2)) * ((bits - 6) / (bits - 3))))

            # If both are over threshold and we have >= 10 genes (double required for single), it is not a false positive 
            if np.all(np.sort(round_freqs)[-2:] > round_freq_threshold) and len(dropped_genes) >= 10: 
                continue
            # if the last round has few bits and the last round has a freq > .5, we have to do different math
            # Still require >= 10 genes
            elif bits % 3 and round_freqs[-1] > .5 and len(dropped_genes) >= 10: 
                final_bits = bits % 3
                final_round_threshold = 0.5 + 0.5 * (1 - (((bits - final_bits - 1) / (bits - 1)) * ((bits - final_bits - 2) / (bits - 2)) * ((bits - final_bits - 3) / (bits - 3))))
            
                if np.max(round_freqs) > round_freq_threshold and np.sort(round_freqs)[-2] > final_round_threshold:
                    continue

            # If none of the above situations are true, we consider the gene a false positive and drop it
            self.fovs.loc[fov, [f'dropout_{gene}' for gene in dropped_genes]] = False
            self.fovs.loc[fov, 'false_positive'] = ";".join(dropped_genes)

    def save_fov_pkl(self, output_dir, filename=""):
        """
        Save the FOV dataframe as a .pkl file
        
        Args:
            output_dir (str/path): the directory in which to store the pkl file
            filename (str) [default='']: if not specified, the file is stored as {experiment_id}_fovs.pkl
        
        Output:
            Saves a pkl file to the filesystem
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError as e:
            pass
            
        if filename == "":
            filename = f'{self.experiment_id}_fovs.pkl'
        else:
            if len(filename.split('.')) == 1:
                filename += ".pkl"
            elif filename.split('.')[-1] != 'pkl':
                raise ValueError("Filename must end in .pkl")
        self.fovs.to_pickle(Path(output_dir) / filename)
        
    def save_fov_tsv(self, output_dir, filename=""):
        """
        Save the FOV dataframe as a .txt.gz file
        
        Args:
            output_dir (str/path): the directory in which to store the .txt.gz file
            filename (str) [default='']: if not specified, the file is stored as {experiment_id}_fovs.txt.gz
        
        Output:
            Saves a pkl file to the filesystem
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError as e:
            pass
        
        if filename == "":
            filename = f'{self.experiment_id}_fovs.txt.gz'
        else:
            if len(filename.split('.')) == 1:
                filename += ".txt.gz"
        self.fovs.to_csv(Path(output_dir) / filename, sep="\t")