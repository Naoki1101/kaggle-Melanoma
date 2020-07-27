import pandas as pd
import numpy as np

# https://www.kaggle.com/cdeotte/jpeg-isic2019-384x384?select=train


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    train2019_df = pd.read_csv('../data/input/train_2019.csv')

    train_df['year'] = 2020
    train2019_df['year'] = 2019

    le = {
        'anterior torso': 'torso', 
        'lower extremity': 'lower extremity', 
        'head/neck': 'head/neck', 
        'upper extremity': 'upper extremity',
        'posterior torso': 'torso', 
        'palms/soles': 'palms/soles', 
        'oral/genital': 'oral/genital', 
        'lateral torso': 'torso'
    }
    train2019_df['anatom_site_general_challenge'] = train2019_df['anatom_site_general_challenge'].map(le)

    use_cols = ['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'target', 'year']
    concat_df = pd.concat([train_df[use_cols], train2019_df[use_cols]], axis=0, sort=False, ignore_index=True)
    concat_df.to_csv('../data/input/train_concated.csv', index=False)


if __name__ == '__main__':
    main()
