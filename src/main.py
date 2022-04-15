import pandas as pd
import os 

def remove_series(serie_paths:list = []) -> bool:
    for path in serie_paths:
        os.remove(path)


def remove_outliers(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    # Remove every sagittal T2 and outliers
    saggital_t2 = list (df_neuro[(df_neuro['ImagePlane'] == 'Sagittal') & 
                           (df_neuro['ImageType'] == 'T2')]['NiftiPath']
                        )
    outliers_desc = list(df_comments[df_comments['Comment'] == 'Outlier']['SeriesDescription'])
    outliers = list(df_neuro[df_neuro['SeriesDescription'].isin(outliers_desc)]['NiftiPath'])

    series2remove = saggital_t2 + outliers
    #remove_series(series2remove)
    df_neuro_new = df_neuro.copy()
    df_neuro_new = df_neuro_new.drop(df_neuro_new[df_neuro_new['NiftiPath'].isin(series2remove)].index)
    return df_neuro_new

def re_label_orientation(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_series2reorient = df_comments[df_comments['Comment'].isin(['Axial', 'Coronal', 'Sagittal'])]['SeriesDescription', 'Comment']
    df_series2reorient
    df_neuro_new = df_neuro.copy()
    df_neuro_new = pd.merge(df_neuro_new, df_series2reorient, on='SeriesDescription', how='inner')
    df_neuro_new = df_neuro_new.drop(columns='ImagePlane')
    df_neuro_new = df_neuro_new.rename({'Comment' : 'ImagePlane'})
    return df_neuro_new

def re_label_t1_ir(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_t1ir = df_comments.copy()
    df_neuro_new = df_neuro.copy()
    cond = (df_t1ir['Comment'] == 'T1 IR')
    series2relabel = list(df_t1ir[cond]['Series Description'])
    for s in series2relabel:
        cond_tmp = df_neuro_new['SeriesDescription'] == s
        df_neuro_new[cond_tmp]['ImageType'] = 'T1 IR'
    return df_neuro_new

def re_label_image_type_OTHER(df_comments:pd.DataFrame, df_neuro:pd.DataFrame) -> pd.DataFrame:
    df_others = df_comments.copy()
    df_neuro_new = df_neuro.copy()
    df_others = df_others[df_others['Image Plane'] == 'OTHER']
    df_others = df_others[['Series Description', 'Comments']]
    df_others = df_others.rename({'Comment':'ImageTypeNew'})
    df_neuro_new = pd.merge(df_neuro_new, df_others, on='SeriesDescription', how='left')
    df_neuro_new['ImageType'] = df_neuro_new.apply(lambda x: x['ImageTypeNew'] if x['ImageTypeNew'] != None else x['ImageType'])
    return df_neuro_new
    
def re_order_database(df_neuro:pd.DataFrame) -> pd.DataFrame:
    def create_new_nifti_path(x):
        x_s = x['NiftiPath'].split('/')[-1]
        x_r = x['NiftiPath'].split('/')[:-1]
        x_t = x['ImageType']
        x_new = '/'.join(x_r, x_t, x_s)
        return x_new

    df_neuro_new = df_neuro.copy()
    df_neuro_new['NewNiftiPath'] = df_neuro_new.apply(create_new_nifti_path)
    for old, new in zip(df_neuro_new['NiftiPath'], df_neuro_new['NewNiftiPath']):
        new_path = new[:-1]
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        os.replace(old, new)
        if os.path.exists(old):
            os.remove(old)
    df_neuro_new['NiftiPath'] = df_neuro_new['NewNiftiPath']
    df_neuro_new.drop(columns='NewNiftiPaht')
    return df_neuro_new

        

if __name__ == '__init__':
    # Read csv
    df_comments = pd.read_csv('~/Tesis/Tesis_Preprocessing/data/limpieza_neuro_db.csv', encoding='cp1252')
    df_comments['Comments'] = df_comments['Comments'].apply(lambda x: x.str.strip())
    df_comments = df_comments.rename({
                                       'Series Description':'SeriesDescrition',
                                       'Image Type': 'ImageType',
                                       'Image Plane': 'ImagePlane'
                                     })
    df_comments['SeriesDescription'] = df_comments['SeriesDescription'].apply(lambda x: x.str.strip())
    df_neuro = pd.read_csv('~/Tesis/Tesis_Preprocessing/data/neuro_db_metadata.csv')
    # Remove every sagittal T2 and outliers
    df_neuro_new = remove_outliers(df_comments, df_neuro)
    # Re-label Orientation
    df_neuro_new = re_label_orientation(df_comments, df_neuro_new)
    # Re-label T1 IR images
    df_neuro_new = re_label_t1_ir(df_comments, df_neuro_new)
    # Re-label Image Type of OTHER
    df_neuro_new = re_label_image_type_OTHER(df_comments, df_neuro_new)
    # Re-order database
    df_neuro_new = re_order_database(df_neuro_new)
    # Save new neuro db
    df_neuro_new.to_csv('~/Tesis/Tesis_Preprocessing/data/neuro_db_metadata_new.csv', sep=',', header=True, index=False)




    
