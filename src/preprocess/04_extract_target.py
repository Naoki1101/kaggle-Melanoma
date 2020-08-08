import pandas as pd


def main():
    train_df = pd.read_csv('../data/input/train_concated.csv')

    train_df[['target']].to_feather('../features/target.feather')


if __name__ == '__main__':
    main()
