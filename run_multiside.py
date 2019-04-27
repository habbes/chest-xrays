import argparse
from baseline_core import run_multiside

if __name__ == '__main__':
    print("Multiside model")
    output_dir = './models/multiside'
    run_multiside(output_dir=output_dir)
