import argparse
from pathlib import Path, PurePath
import os
import subprocess

def main(barcodes, codebook_path, output_dir):
    # Find ourt current path
    if PurePath(__file__).is_absolute():
        script_dir = Path(__file__).parent
    else:
        script_dir = (Path.cwd() / Path(__file__)).parent

    # Run QC on all barcodes in the list
    dropout_jobs = []
    for barcode in barcodes.split(','):
        python_command = f'python {script_dir / "dropout_QC_script.py"} --barcode {barcode} --codebook {codebook_path} --output_dir {output_dir / barcode}'
        dropout_jobs.append(submit_slurm_job(f'dropout_qc_{barcode}', output_dir / barcode, python_command))
        
    # Do some analysis on all the barcodes in the list together
    python_command = f'python {script_dir / "dropout_QC_post_slurm.py"} --barcodes {barcodes} --output_dir {output_dir}'
    all_jobs = dropout_jobs + [submit_slurm_job(f'dropout_qc_analysis', output_dir, python_command, memory="--mem=100G", dependency=f"--dependency=afterok:{','.join(dropout_jobs)}")]

    print("To view slurm jobs:")
    print(f"sacct --format=JobID%30,JobName%30,Partition,Account,User,State,Elapsed,TimeLimit -j {','.join(all_jobs)}")
    
def submit_slurm_job(job_name, output_dir, python_command, partition="--partition=celltypes", memory="--mem=30G", time="--time 02:00:00", dependency=""):
    job_out = output_dir / (job_name + ".out")
    job_err = output_dir / (job_name + ".err")

    # Creating the out and err files for the batch job
    try:
        os.makedirs(output_dir)
    except FileExistsError as e:
        pass
    if job_out.exists():
        os.remove(job_out)
    if job_err.exists():
        os.remove(job_err)
    try:
        job_out.touch()
        job_err.touch()
    except FileExistsError as err:
        # This error should never occur because we are deleting the files first
        print(err)
        return

    # Create a string for the slurm command
    slurm_command = "sbatch {} --job-name={}.job --output={} --error={} {} {} {} --wrap='{}'".format(dependency, job_name, job_out, job_err, partition, memory, time, python_command)

    sp = subprocess.run(slurm_command, shell=True, check=True, universal_newlines=True, stdout=subprocess.PIPE)

    if not sp.stdout.startswith("Submitted batch"):
        raise ChildProcessError("SlurmError: sbatch not submitted correctly")

    return sp.stdout.split()[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--barcodes', type=str, help='comma seperated list of barcodes to run QC on')
    parser.add_argument('--codebook', type=str, help='codebook to use for false positive correction (only supports one codebook for ALL barcodes)')
    parser.add_argument('--output_dir', type=str, help='directory to output results into (subdirectories will be made for each barcode)')
    
    config = parser.parse_args()
    main(config.barcodes, config.codebook, Path(config.output_dir))