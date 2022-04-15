import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from PIL import Image

ROOT_PATH = '/mnt/d/Tesis'

def show_images(slice:np.array, title:str) -> None:
    plt.figure(figsize=(10,8))
    fig, axes = plt.subplots(1,1,figsize=(10, 8))
    axes.imshow(slice, cmap='gray')
    axes.set_title(title)
    plt.show()

def show_image_PIL(slice:np.array)->None:
    image = Image.fromarray(slice)
    image.show()


def main(args):
    df = pd.read_csv('~/Tesis/Tesis_Preprocessing/data/neuro_db_metadata.csv')
    df = df[(df['ImagePlane'] == args.image_plane) & 
            (df['ImageType'] == args.image_type)]
    df = df[['SeriesDescription', 'NiftiPath']]
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    SeriesDescriptions = df['SeriesDescription'].unique()
    for s in SeriesDescriptions:
        path = df[df['SeriesDescription'] == s]['NiftiPath'].iloc[0]
        vol = nib.load(os.path.join(ROOT_PATH, path)).get_fdata()
        if args.image_plane == 'Axial':
            z_slices = vol.shape[-1]
            slice_num = int(z_slices // 2)
            _slice = vol[:,:,slice_num]
        if args.image_plane == 'Sagittal':
            y_slices = vol.shape[1]
            slice_num = int(y_slices // 2)
            _slice = vol[:,slice_num,:]
        if args.image_plane == 'Coronal':
            x_slices = vol.shape[0]
            slice_num = int(x_slices // 2)
            _slice = vol[slice_num,:,:]
        print(s)
        print(path)
        show_image_PIL(_slice)
        input()


def manage_arguments():
    parser = ArgumentParser()
    parser.add_argument('--image_plane',
                        default='Axial',
                        required=False,
                        metavar='image_plane')
    parser.add_argument('--image_type', 
                        default='T1',
                        required=False,
                        metavar='image_type')
    return parser.parse_args()

if __name__ == "__main__":
    args = manage_arguments()
    main(args)