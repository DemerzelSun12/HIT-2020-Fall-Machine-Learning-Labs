import pandas as pd
import numpy as np


class ReadBankNote(object):

    def __init__(self):
        self.data_set = pd.read_csv("../data/data_banknote_authentication.csv")
        self.X = self.data_set.drop('class', axis=1)
        self.Y = self.data_set['class']

    def generate_data(self):
        return np.array(self.X, dtype=float), np.array(self.Y, dtype=int)


# if __name__ == '__main__':
#     br = ReadBankNote()
#     print(br.generate_data)
#     print(br.X.head())
#     print(br.Y.head())
#     print(br.X)
#     print(br.Y)
