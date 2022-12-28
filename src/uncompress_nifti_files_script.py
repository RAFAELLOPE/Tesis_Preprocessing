import glob
import gzip 
import shutil
import os
from argparse import ArgumentParser

def uncompress_gzip(org_file:str, dst_file:str) -> bool:
    with gzip.open(org_file, 'rb') as f_in:
        with open(dst_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def manage_arguments():
    parser = ArgumentParser()
    parser.add_argument('--org_path',
                        help='Path containing gzip files',
                        type=str,
                        required=True,
                        default='')
    parser.add_argument('--dst_path',
                        help='Path that will contain uncompressed files',
                        type=str,
                        required=True,
                        default='')
    args=parser.parse_args()
    return args

def main(args):
    org_base_path = args.org_path
    dst_base_path = args.dst_path
    #org_base_path = '/mnt/d/Tesis/neuro_db'
    #dst_base_path = '/mnt/d/Tesis/neuro_db_T1'
    org_paths = glob.glob(os.path.join(org_base_path, '*/*/*/T1/*.nii.gz'))

    for org_path in org_paths:
        dst_pt = '/'.join(org_path.split('/')[:-1]).replace(org_base_path, dst_base_path)
        dst_f = org_path.split('/')[-1].replace('.nii.gz', '.nii')
        
        if not os.path.exists(dst_pt):
            os.makedirs(dst_pt)
        
        dst_path = os.path.join(dst_pt, dst_f)
        uncompress_gzip(org_file=org_path, dst_file=dst_path)

if __name__ == '__main__':
    args = manage_arguments()
    main(args)