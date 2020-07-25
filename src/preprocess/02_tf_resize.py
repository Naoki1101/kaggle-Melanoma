import cv2
import pandas as pd
from tqdm import tqdm
import multiprocessing
import pydicom as dicom
from pydicom.pixel_data_handlers.util import convert_color_space

size_list = [256, 384, 512]


def read_image(id_):
    ds = dicom.dcmread(f'../data/input/train/{id_}.dcm')
    dicom_arr = convert_color_space(ds.pixel_array, "YBR_FULL_422", "RGB")
    return dicom_arr


def resize(data_name, id_, size):
    img = read_image(id_)
    img = cv2.resize(img, dsize=(size, size))
    cv2.imwrite(f'../data/input/tf_resized_{size}/{data_name}/{id_}.png', img)


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    test_df = pd.read_csv('../data/input/test.csv')

    for size in size_list:
        for data_name, df in zip(['train', 'test'], [train_df, test_df]):
            for id_ in tqdm(df['image_name']):
                p = multiprocessing.Process(target=resize, args=(data_name, id_, size))
                p.start()


if __name__ == '__main__':
    main()
