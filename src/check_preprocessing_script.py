import glob as glob
from argparse import ArgumentParser
import os

def manage_arguments():
    parser = ArgumentParser()
    parser.add_argument('--db_path',
                        help='Path to specify the database',
                        type=str,
                        required=True,
                        default='')
    args=parser.parse_args()
    return args

def write_output(l, filename='files_without_segmentation.txt'):
    with open("/home/rlopez/" + filename, "w") as f:
        for line in l:
            f.write(line)
            f.write('\n')

def main(args):
    files_without_segmentation = []
    files_with_segmentation = []
    neuro_db_path = args.db_path
    nifti_files = [item for item in glob.glob(os.path.join(neuro_db_path, '*', '*', '*', 'T1', '*.nii')) 
                  if not item.split('/')[-1].startswith('c')]
    for p in nifti_files:
        split_path = p.split('/')
        f = split_path[-1]
        base_path = '/'.join(split_path[:-1])
        for i in range(1, 6):
            if not os.path.exists(os.path.join(base_path, 'c' + str(i) + f)):
                print(p)
                files_without_segmentation.append(p)
                break
    
    files_with_segmentation = list(set(nifti_files).difference(set(files_without_segmentation)))
    write_output(files_without_segmentation)
    write_output(files_with_segmentation, filename='files_with_segmentation.txt')


if __name__ == "__main__":
    args = manage_arguments()
    main(args)
