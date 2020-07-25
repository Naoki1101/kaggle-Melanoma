import cv2
import pandas as pd
from tqdm import tqdm
import multiprocessing
import pydicom as dicom
from pydicom.pixel_data_handlers.util import convert_color_space

size_list = [256, 384, 512]


def resize(dataset, data_name, id_, size):
    img = filter_tfrecords(dataset, id_)
    img = cv2.resize(img, dsize=(size, size))
    cv2.imwrite(f'../data/input/tf_resized_{size}/{data_name}/{id_}.png', img)


def mapping_func(serialized_example):
    TFREC_FORMAT = {
                    "image_name": tf.io.FixedLenFeature([], tf.string),
                    "image": tf.io.FixedLenFeature([], tf.string)
                    }
    example = tf.io.parse_single_example(serialized_example, features=TFREC_FORMAT)
    return example

def filter_tfrecords(dataset, image_name):
    record_dataset = dataset.filter(lambda example: tf.equal(example["image_name"], image_name))
    example = next(iter(record_dataset))
    arr = tf.image.decode_jpeg(example['image'], channels=3).numpy()
    return arr


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    test_df = pd.read_csv('../data/input/test.csv')

    for size in size_list:
        for data_name, df in zip(['train', 'test'], [train_df, test_df]):
            dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(f'../data/input/tfrecords/{data_name}*.tfrec'))
            dataset = dataset.map(mapping_func)

            for id_ in tqdm(df['image_name']):
                p = multiprocessing.Process(target=resize, args=(dataset, data_name, id_, size))
                p.start()


if __name__ == '__main__':
    main()
