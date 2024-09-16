# ChronoTrader: Predicting Google Stock Prices Using Recurrent Neural Networks

This project demonstrates the application of a Recurrent Neural Network (RNN) to predict Google's stock prices in 2017 based on previous stock data. The model leverages Long Short-Term Memory (LSTM) layers to capture time dependencies within stock price data and produce a prediction.

## Project Overview

The goal of this project is to predict the stock price of Google in 2017 using historical stock prices. An RNN is trained on a sequence of stock prices and used to forecast the stock price for future days. This project helps in understanding how deep learning can be applied to time-series data, like stock prices.

## Problem Solved

Stock prices exhibit complex patterns due to market fluctuations. Predicting future stock prices based on historical data can assist investors in making informed decisions. This project aims to provide a model that helps predict future prices based on past data, even in fluctuating markets.

## Key Features

- **Recurrent Neural Network (RNN):** Utilizes LSTM layers to capture time dependencies and trends within stock price data.
- **Time-Series Data Processing:** Historical stock prices are preprocessed and structured into sequences to train the model.
- **Dropout Regularization:** Dropout is employed to prevent overfitting and ensure better generalization to unseen data.
- **Prediction Performance:** The model forecasts future prices by learning patterns from past trends, achieving reasonable predictions.

## Model Architecture

The RNN model was constructed using the following components:
- **LSTM Layers:** Four LSTM layers with 50 units each are used to capture long-term dependencies in stock price data.
- **Dropout Layers:** Dropout layers with a rate of 0.2 are added after each LSTM layer to prevent overfitting.
- **Dense Layer:** A fully connected layer with a single output unit to predict the stock price for the next time step.
- **Optimizer:** The Adam optimizer is used with a mean squared error loss function.

## Results

After training the RNN for 100 epochs with a batch size of 32, the following results were observed:
- The model was able to capture the general trend of stock price movements.
- Below is a visualization of the real stock price vs. the predicted stock price:

![Stock Price Prediction](results/predicted_stock_price_vs_actual_stock_price.png)

## Future Improvements

Several enhancements can be made to improve the model's performance:
- **Data Augmentation:** Incorporating additional stock data such as trading volume and external economic factors could improve accuracy.
- **Hyperparameter Tuning:** Further tuning of batch size, learning rate, and number of LSTM units might improve prediction results.
- **Advanced Architectures:** Explore more advanced architectures like Bidirectional LSTMs and GRUs for better time-series forecasting.
- **Transfer Learning:** Utilize pre-trained models to enhance the generalization capabilities of the network.

## Dataset

(IMPORTANT: Data Files were uploaded to Google Drive before it was mounted).

import pandas as pd

# Load the training and testing data from the local 'data/' folder
train_file_path = 'data/Google_Stock_Price_Train.csv'
test_file_path = 'data/Google_Stock_Price_Test.csv'

# Read the training data
dataset_train = pd.read_csv(train_file_path)
training_set = dataset_train.iloc[:, 1:2].values  # Assuming 'Open' is the second column

# Read the testing data
dataset_test = pd.read_csv(test_file_path)
test_set = dataset_test.iloc[:, 1:2].values  # Assuming 'Open' is the second column

# License

MIT License

Copyright (c) [Year] [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

