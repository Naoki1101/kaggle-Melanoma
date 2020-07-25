import cv2
import pandas as pd
import multiprocessing
from tqdm import tqdm

size_list = [256, 384, 512]


def resize(data_name, id_, size):
    img = cv2.imread(f'../data/input/jpeg/{data_name}/{id_}.jpg')
    img = cv2.resize(img, dsize=(size, size))
    cv2.imwrite(f'../data/input/jpeg_resized_{size}/{data_name}/{id_}.png', img)


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
