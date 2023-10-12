"""
This file is for developing algorithm and predicting House price based on training and testing the model
"""

# import numpy as np
import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class HPP:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)
        self.model = LinearRegression()
        # print(self.df)


    def split_data(self):
        try:
            # Give independent data to x & dependent data to y
            x = self.df.iloc[:, 1:]                # independent (all row, 1st to all col)
            y = self.df.iloc[:, 0]                 # dependent (all row, 0th col)
            # Splitting the data using train_test method
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            # test_size indicates percentage of values for test i.e, 20% Data = Test & 80% Data = Train
            return x_train, x_test, y_train, y_test
        except Exception as E:
            print(f'Error in main - {E.__str__()}')

    def model_training(self, x_train, y_train):
        try:
            # Now, giving training data to train the algorithm
            self.model.fit(x_train, y_train)
            y_train_pred = self.model.predict(x_train)
            # r2_score = used to find accuracy of the model
            # find accuracy of train_data
            print(f'Train accuracy : {r2_score(y_train, y_train_pred)}')
            # find loss of train_data (Formula: loss = 1 - accuracy)
            print(f'Train loss : {1 - r2_score(y_train, y_train_pred)}')
        except Exception as E:
            print(f'Error in main - {E.__str__()}')

    def model_testing(self, x_test, y_test):
        try:
            y_test_pred = self.model.predict(x_test)
            # find accuracy of test_data
            print(f'Test accuracy : {r2_score(y_test, y_test_pred)}')
            # find loss of test_data
            print(f'Test loss : {1 - r2_score(y_test, y_test_pred)}')
        except Exception as E:
            print(f'Error in main - {E.__str__()}')

    def preprossing(self):
        try:
            # print(self.df.isnull().sum())  # find null values
            # print(f'No of rows and no of col in Dataset: {self.df.shape[0], self.df.shape[1]}')
            # converting cat data to numerical data using Map method
            self.df['mainroad'] = self.df['mainroad'].map({'yes': 1, 'no': 0})
            self.df['guestroom'] = self.df['guestroom'].map({'yes': 1, 'no': 0})
            self.df['basement'] = self.df['basement'].map({'yes': 1, 'no': 0})
            self.df['hotwaterheating'] = self.df['hotwaterheating'].map({'yes': 1, 'no': 0})
            self.df['airconditioning'] = self.df['airconditioning'].map({'yes': 1, 'no': 0})
            self.df['furnishingstatus'] = self.df['furnishingstatus'].map(
                {'furnished': 0, 'semi-furnished': 1, 'unfurnished': 2})
            print(f'{self.df.head()}')

            x_train, x_test, y_train, y_test = self.split_data()
            print(f'x_train shape : {x_train.shape} y_train shape : {y_train.shape}')
            print(f'x_test shape : {x_test.shape} y_test shape : {y_test.shape}')
            self.model_training(x_train, y_train)
            self.model_testing(x_test, y_test)
        except Exception as E:
            print(f'Error in main - {E.__str__()}')


if __name__ == '__main__':  # main mathod (main processor)
    try:
        obj = HPP('J:/Courses/Vihara tech (Internship)/Projects/Pro1_House Price Pred/Housing.csv')
        # use forward slash (clien's requirement)
        # create instance(object) and do operations
        # for preprossing will use a function
        obj.preprossing()
    except Exception as E:
        print(f'Error in main - {E.__str__()}')
