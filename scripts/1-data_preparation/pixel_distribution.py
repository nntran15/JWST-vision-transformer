#!/usr/bin/env python3
"""
Script to analyze dimensions of FITS files and create visualization plots.

This script processes a directory containing FITS files and generates:
1. A scatterplot of x-dimension vs y-dimension
2. An 11x11 heatmap matrix showing file count distribution across dimension bins
"""

import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_fits_dimensions(fits_path):
    """
    Extract the dimensions and instrument of a FITS file.
    
    Args:
        fits_path: Path to the FITS file
        
    Returns:
        tuple: (fits_path, x_dim, y_dim, instrument) where instrument
               is the INSTRUME header value (e.g. 'NIRCAM', 'MIRI') or None
    """
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            # Read instrument from the first HDU header
            instrument = hdul[0].header.get('INSTRUME', None)
            # Try to get data from the first HDU with data
            for hdu in hdul:
                if hdu.data is not None:
                    # FITS shape is (y, x) in NumPy convention
                    shape = hdu.data.shape
                    if len(shape) >= 2:
                        y_dim, x_dim = shape[-2], shape[-1]
                        return (str(fits_path), x_dim, y_dim, instrument)
        return (str(fits_path), None, None, None)
    except Exception as e:
        # Silently skip files that can't be read
        return (str(fits_path), None, None, None)


def assign_bin(dimension):
    """
    Assign a dimension value to its corresponding bin index (0-10).
    
    Bins: 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80, 81-90, 91-100, 100+
    
    Args:
        dimension: The dimension value
        
    Returns:
        int: Bin index (0-10)
    """
    if dimension <= 10:
        return 0
    elif dimension <= 100:
        return (dimension - 1) // 10
    else:
        return 10


def create_scatterplot(x_dims, y_dims, output_path):
    """
    Create a scatterplot of x-dimension vs y-dimension.
    
    Args:
        x_dims: List of x-dimensions
        y_dims: List of y-dimensions
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Use alpha for transparency to handle overlapping points
    plt.scatter(x_dims, y_dims, alpha=0.3, s=1, c='blue')
    
    plt.xlabel('X-Dimension (pixels)', fontsize=12)
    plt.ylabel('Y-Dimension (pixels)', fontsize=12)
    plt.title('FITS File Dimensions Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatterplot saved to: {output_path}")


def create_heatmap(dimension_matrix, output_path):
    """
    Create an 11x11 heatmap matrix showing file count distribution.
    
    Args:
        dimension_matrix: 11x11 numpy array with file counts
        output_path: Path to save the plot
    """
    # Define bin labels
    bin_labels = [
        '0-10', '11-20', '21-30', '31-40', '41-50', '51-60',
        '61-70', '71-80', '81-90', '91-100', '100+'
    ]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap with logarithmic scale for better visualization
    log_matrix = np.log10(dimension_matrix + 1)
    
    im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(11))
    ax.set_yticks(np.arange(11))
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_yticklabels(bin_labels, fontsize=10)
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations showing actual counts
    for i in range(11):
        for j in range(11):
            count = int(dimension_matrix[i, j])
            if count > 0:
                # Use white text for darker cells, black for lighter cells
                text_color = 'white' if log_matrix[i, j] > log_matrix.max() / 2 else 'black'
                text = ax.text(j, i, f'{count:,}',
                             ha='center', va='center', color=text_color, fontsize=8)
    
    # Add colorbar with log scale indication
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Log10(File Count + 1)', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_xlabel('Y-Dimension Bins (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('X-Dimension Bins (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('FITS File Dimension Distribution Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {output_path}")


def process_fits_directory(directory_path, output_dir, max_workers=None, instrument=None):
    """
    Process all FITS files in a directory and generate visualizations.
    
    Args:
        directory_path: Path to directory containing FITS files
        output_dir: Directory to save output plots
        max_workers: Number of processes/workers to use (default: CPU count)
        instrument: If specified, only count files from this JWST instrument
                    (e.g. 'NIRCAM', 'MIRI', 'NIRISS', 'NIRSPEC', 'FGS')
    """
    if instrument:
        instrument = instrument.upper()
        print(f"Filtering by instrument: {instrument}")
    directory = Path(directory_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all FITS files
    print(f"Searching for FITS files in: {directory}")
    fits_files = list(directory.glob('**/*.fits'))
    
    if not fits_files:
        print("No FITS files found!")
        return
    
    print(f"Found {len(fits_files)} FITS files. Processing...")
    
    # Lists to store dimensions
    x_dims = []
    y_dims = []
    
    # 11x11 matrix for binned counts
    dimension_matrix = np.zeros((11, 11), dtype=int)
    
    # Process each FITS file
    failed_count = 0
    skipped_count = 0
    instrument_skipped = 0
    
    # We use chunks for processing progress reporting and memory efficiency
    chunksize = max(1, len(fits_files) // (max_workers * 10 if max_workers else os.cpu_count() * 10))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit map jobs
        futures = executor.map(get_fits_dimensions, fits_files, chunksize=chunksize)
        
        for fits_path, x_dim, y_dim, file_instrument in tqdm(futures, total=len(fits_files), desc="Processing FITS files"):
            if x_dim is not None and y_dim is not None:
                # Filter by instrument if specified
                if instrument and (file_instrument is None or file_instrument.upper() != instrument):
                    instrument_skipped += 1
                    continue
                if 20 <= x_dim <= 100 and 20 <= y_dim <= 100:
                    x_dims.append(x_dim)
                    y_dims.append(y_dim)
                    
                    # Assign to bins
                    x_bin = assign_bin(x_dim)
                    y_bin = assign_bin(y_dim)
                    dimension_matrix[x_bin, y_bin] += 1
                else:
                    skipped_count += 1
            else:
                failed_count += 1
    
    print(f"\nProcessed {len(x_dims)} files successfully")
    if instrument_skipped > 0:
        print(f"Skipped {instrument_skipped} files (different instrument)")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files due to dimension cutoffs")
    if failed_count > 0:
        print(f"Failed to read {failed_count} files")
    
    # Generate statistics
    print(f"\nDimension Statistics:")
    print(f"  X-dimension: min={min(x_dims)}, max={max(x_dims)}, mean={np.mean(x_dims):.1f}")
    print(f"  Y-dimension: min={min(y_dims)}, max={max(y_dims)}, mean={np.mean(y_dims):.1f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    scatterplot_path = output_dir / 'dimension_scatterplot.png'
    create_scatterplot(x_dims, y_dims, scatterplot_path)
    
    heatmap_path = output_dir / 'dimension_heatmap.png'
    create_heatmap(dimension_matrix, heatmap_path)
    
    print("\nDone!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Analyze FITS file dimensions and generate visualization plots.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/fits/directory
  %(prog)s /path/to/fits/directory --output ./plots
        """
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing FITS files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory for plots (default: ./output)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=os.cpu_count(),
        help='Number of worker processes to use (default: CPU count)'
    )
    
    parser.add_argument(
        '--instrument', '-i',
        type=str,
        default=None,
        help='Only count thumbnails from this JWST instrument (e.g. NIRCAM, MIRI, NIRISS, NIRSPEC, FGS). Case-insensitive.'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.directory).exists():
        print(f"Error: Directory '{args.directory}' does not exist!", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.directory).is_dir():
        print(f"Error: '{args.directory}' is not a directory!", file=sys.stderr)
        sys.exit(1)
    
    # Process the directory
    process_fits_directory(args.directory, args.output, max_workers=args.workers, instrument=args.instrument)


if __name__ == '__main__':
    main()
