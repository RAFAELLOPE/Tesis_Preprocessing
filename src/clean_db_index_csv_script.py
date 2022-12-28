import pandas as pd
import os
import glob as glob
import nibabel as nib
from argparse import ArgumentParser

def remove_series(serie_paths:list = []) -> bool:
    for path in serie_paths:
        path2remove = os.path.join('/mnt/d/Tesis', path)
        if os.path.exists(path2remove):
            os.remove(path2remove)


def remove_outliers(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    # Remove every sagittal T2 and outliers
    saggital_t2 = list (df_neuro[(df_neuro['ImagePlane'] == 'Sagittal') & 
                           (df_neuro['ImageType'] == 'T2')]['NiftiPath']
                        )
    outliers_desc = list(df_comments[df_comments['Comment'] == 'OUTLIER']['SeriesDescription'])
    outliers = list(df_neuro[df_neuro['SeriesDescription'].isin(outliers_desc)]['NiftiPath'])

    series2remove = saggital_t2 + outliers
    remove_series(series2remove)
    df_neuro_new = df_neuro.copy()
    df_neuro_new = df_neuro_new.drop(df_neuro_new[df_neuro_new['NiftiPath'].isin(series2remove)].index)
    df_neuro_new = df_neuro_new.reset_index(drop=True)
    return df_neuro_new

def re_label_orientation(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_series2reorient = df_comments[df_comments['Comment'].isin(['AXIAL', 'CORONAL', 'SAGITTAL'])][['SeriesDescription', 'Comment']]
    df_series2reorient = df_series2reorient.rename(columns={'Comment':'NewImagePlane'})
    df_neuro_new = df_neuro.copy()
    df_neuro_new = pd.merge(df_neuro_new, df_series2reorient, on='SeriesDescription', how='left')
    df_neuro_new['ImagePlane'] = df_neuro_new.apply(lambda x: x['NewImagePlane'] if not pd.isna(x['NewImagePlane']) else x['ImagePlane'], axis=1)
    df_neuro_new = df_neuro_new.drop(columns='NewImagePlane')
    return df_neuro_new

def re_label_t1_ir(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_t1ir = df_comments.copy()
    df_neuro_new = df_neuro.copy()
    cond = (df_t1ir['Comment'] == 'T1 IR')
    series2relabel = list(df_t1ir[cond]['SeriesDescription'])
    for s in series2relabel:
        cond_tmp = df_neuro_new['SeriesDescription'] == s
        df_neuro_new.loc[cond_tmp,'ImageType'] = 'T1 IR'
    return df_neuro_new

def re_label_image_type_OTHER(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_others = df_comments.copy()
    df_neuro_new = df_neuro.copy()
    df_others = df_others[df_others['ImageType'] == 'OTHER']
    df_others = df_others[['SeriesDescription', 'Comment']]
    df_others = df_others.rename(columns={'Comment':'ImageTypeNew'})
    df_neuro_new = pd.merge(df_neuro_new, df_others, on='SeriesDescription', how='left')
    df_neuro_new['ImageType'] = df_neuro_new.apply(lambda x: x['ImageTypeNew'] if not pd.isna(x['ImageTypeNew']) else x['ImageType'], axis=1)
    df_neuro_new = df_neuro_new.drop(columns='ImageTypeNew')
    return df_neuro_new
    
def re_order_database(df_neuro:pd.DataFrame) -> pd.DataFrame:
    def create_new_nifti_path(x):
        x_s = x['NiftiPath'].split('/')[-1]
        x_r = '/'.join(x['NiftiPath'].split('/')[:-1])
        x_t = x['ImageType']
        x_new = os.path.join(x_r, x_t, x_s)
        return x_new

    df_neuro_new = df_neuro.copy()
    df_neuro_new['NewNiftiPath'] = df_neuro_new.apply(create_new_nifti_path, axis=1)
    for old, new in zip(df_neuro_new['NiftiPath'], df_neuro_new['NewNiftiPath']):
        new_folder = os.path.join('/mnt/d/Tesis', '/'.join(new.split('/')[:-1]))
        new_path = os.path.join('/mnt/d/Tesis', new)
        old_path = os.path.join('/mnt/d/Tesis', old)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        os.replace(old_path, new_path)
        if os.path.exists(old_path):
            os.remove(old_path)
    df_neuro_new['NiftiPath'] = df_neuro_new['NewNiftiPath']
    df_neuro_new.drop(columns='NewNiftiPath')
    return df_neuro_new

def remove_not_labeled_OTHER(df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_neuro_copy = df_neuro.copy()
    df_other2remove = df_neuro_copy[df_neuro_copy['ImageType'] == 'OTHER']
    for path in df_other2remove['NiftiPath']:
        path2remove = os.path.join('/mnt/d/Tesis', path)
        if os.path.exists(path2remove):
            os.remove(path2remove)
    df_neuro_copy = df_neuro_copy.drop(df_neuro_copy[df_neuro_copy['ImageType'] == 'OTHER'].index)
    df_neuro_copy = df_neuro_copy.reset_index(drop=True)
    return df_neuro_copy

def remove_bval_bvec_files() -> None:
    bval_files = glob.glob(os.path.join('/mnt/d/Tesis/neuro_db', '*', '*', '*', '*.bval'))
    bvec_files = glob.glob(os.path.join('/mnt/d/Tesis/neuro_db', '*', '*', '*', '*.bvec'))
    files2remove = list(bval_files) + list(bvec_files)
    for f in files2remove:
        if os.path.exists(f):
            os.remove(f)

def remove_series_with_low_slices(df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_neuro_copy = df_neuro.copy()
    for path in df_neuro_copy['NiftiPath']:
        path2check = os.path.join('/mnt/d/Tesis', path)
        if os.path.exists(path2check):
            vol = nib.load(path2check).get_fdata()
            shape = vol.shape
            if min(shape) < 20:
                # Remove the series
                print(f'Removing series {path} ...')
                df_neuro_copy = df_neuro_copy.drop(df_neuro_copy[df_neuro_copy['NiftiPath'] == path].index)
                #os.remove(path2check)
    return df_neuro_copy

def remove_empty_folders()-> None:
    folders2check = glob.glob(os.path.join('/mnt/d/Tesis/neuro_db', '*', '*', '*', '*'))
    for f in folders2check:
        if not f.endswith('txt'):
            if len(os.listdir(f)) == 0:
                print(f'Removing empty directory: {f}')
                os.rmdir(f)

def manage_arguments():
    parser = ArgumentParser()
    parser.add_argument('--neuro_db_file',
                        help='Path where input file with neuro db data is located',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--cleaning_file',
                        help='Path where input file with cleaning actions is located',
                        type=str,
                        required=True,
                        default='')
    args=parser.parse_args()
    return args

def main(args):
    neuro_db_csv_path = args.neuro_db_file
    cleaing_csv_path = args.cleaning_file
    
    assert os.path.exists(neuro_db_csv_path)
    assert os.path.exists(cleaing_csv_path)

    # Read csv
    df_comments = pd.read_csv('~/Tesis/Tesis_Preprocessing/data/limpieza_neuro_db.csv', encoding='cp1252')
    df_comments['Comment'] = df_comments['Comment'].apply(lambda x: str(x).strip().upper())
    df_comments = df_comments.dropna()
    df_comments = df_comments.rename({
                                       'Series Description':'SeriesDescrition',
                                       'Image Type': 'ImageType',
                                       'Image Plane': 'ImagePlane'
                                     })
    df_comments['SeriesDescription'] = df_comments['SeriesDescription'].apply(lambda x: str(x).strip())
    df_neuro = pd.read_csv(os.path.join(neuro_db_csv_path, 'neuro_db_metadata_new.csv'))
    df_neuro_new = remove_series_with_low_slices(df_neuro)
    # Remove every sagittal T2 and outliers
    df_neuro_new = remove_outliers(df_comments, df_neuro)
    # Re-label Orientation
    df_neuro_new = re_label_orientation(df_comments, df_neuro_new)
    # Re-label T1 IR images
    df_neuro_new = re_label_t1_ir(df_comments, df_neuro_new)
    # Re-label Image Type of OTHER
    df_neuro_new = re_label_image_type_OTHER(df_comments, df_neuro_new)
    # Remove Image Types OTHER not labeled before
    df_neuro_new = remove_not_labeled_OTHER(df_neuro_new)
    # Remove bval and bvec files
    remove_bval_bvec_files()
    # Re-order database
    df_neuro_new = re_order_database(df_neuro_new)
    # Remove empty folders
    remove_empty_folders()
    # Save new neuro db
    df_neuro_new['ImageType'] = df_neuro_new['ImageType'].apply(lambda x: str(x).upper())
    df_neuro_new['ImagePlane'] = df_neuro_new['ImagePlane'].apply(lambda x: str(x).upper())
    df_neuro_new.to_csv('~/Tesis/Tesis_Preprocessing/data/neuro_db_metadata_new.csv', sep=',', header=True, index=False)

if __name__ == '__main__':
    args = manage_arguments()
    main(args)

    
