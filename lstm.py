import pandas as pd
from keras import Sequential
from tensorflow import keras

NUM_OUTPUT = 2


def create_model(num_outputs, **kwargs):
    model = keras.layers.LSTM(units=num_outputs,
                              activation="tanh",
                              recurrent_activation="sigmoid",
                              use_bias=True,
                              kernel_initializer="glorot_uniform",
                              recurrent_initializer="orthogonal",
                              bias_initializer="zeros",
                              unit_forget_bias=True,
                              kernel_regularizer=None,
                              recurrent_regularizer=None,
                              bias_regularizer=None,
                              activity_regularizer=None,
                              kernel_constraint=None,
                              recurrent_constraint=None,
                              bias_constraint=None,
                              dropout=0.0,
                              recurrent_dropout=0.0,
                              return_sequences=False,
                              return_state=False,
                              go_backwards=False,
                              stateful=False,
                              time_major=False,
                              unroll=False,
                              **kwargs)

    return model

def train_test_split():

if __name__ == '__main__':
    # Read data set
    data = pd.read_csv("")
    model = Sequential()
    model.add(create_model(200))
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dense(NUM_OUTPUT))
    model.compile()

    model.fit()
