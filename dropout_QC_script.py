import argparse
from dropout_detection import TranscriptImage
from pathlib import Path
import os

def main(barcode, codebook_path, output_dir):
    mask_out_dir = output_dir / 'tissue_masks'

    # Find dropped FOVs and save them
    ts = TranscriptImage(barcode)
    ts.run_dropout_pipeline(mask_out_dir, codebook_path, threshold=0.15)
    ts.save_fov_pkl(output_dir)
    ts.save_fov_tsv(output_dir)

    if ts.get_considered_fov_counts() > 0:
        # Write results to file
        with open(output_dir / 'dropout_summary.txt', 'w+') as f:
            f.write(ts.dropout_summary(return_summary=True))
        
        # Draw results
        try:
            os.makedirs(output_dir / 'images')
        except FileExistsError as e:
            pass
        ts.draw_total_dropout_map(out_file=output_dir / 'images/total_dropout_map.png')
        ts.draw_dropped_genes(output_dir / 'images')
    else:
        # Write results to file
        with open(output_dir / 'dropout_summary.txt', 'w+') as f:
            f.write("There were no FOVs determined to be on tissue for this experiment. This is likely due to low transcript density.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--barcode', type=str, help='barcode to run QC on')
    parser.add_argument('--codebook', type=str, help='codebook to use for false positive correction')
    parser.add_argument('--output_dir', type=str, help='directory to output results into')
    
    config = parser.parse_args()
    main(config.barcode, config.codebook, Path(config.output_dir))
