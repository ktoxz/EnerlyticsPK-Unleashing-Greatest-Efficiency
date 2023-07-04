# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import tensorflowjs as tfjs

# #Load the training file and label
# df = pd.read_csv('Cleaned_Electricity_Data.csv')
# df = df.drop(columns = ['index', 'Date','Time'])
# ytrain = df.drop(columns=['Voltage', 'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
# df = df.drop(columns = ['Global_reactive_power','Global_active_power'])

# #Load the validation file and label
# dfeval = pd.read_csv('PredictingValue.csv')
# dfeval = dfeval.drop(columns = ['index', 'Date','Time'])
# yeval = dfeval.drop(columns=['Voltage', 'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
# dfeval = dfeval.drop(columns = ['Global_reactive_power','Global_active_power'])

# #build the Sequential model with 2 hidden layers. The first layer has 5 nodes, 2 hidden layers have 10 nodes and the output layer has 2 nodes
# def build_model(my_learning_rate):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(units=10,
#                                   input_shape=(5,)))
#     model.add(tf.keras.layers.Dense(units = 10))
#     model.add(tf.keras.layers.Dense(units=2))
#     model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
#                 loss="mean_squared_error",
#                 metrics=[tf.keras.metrics.RootMeanSquaredError()])
   
#     return model

# def train_model(model, feature, label, epochs, batch_size):
#     history = model.fit(x=feature,
#                       y=label,
#                       batch_size=batch_size,
#                       epochs=epochs)
#     trained_weight = model.get_weights()[0]
#     trained_bias = model.get_weights()[1]
#     epochs = history.epoch
#     hist = pd.DataFrame(history.history)
#     rmse = hist["root_mean_squared_error"]
#     return trained_weight, trained_bias, epochs, rmse

# learning_rate=0.005
# epochs=20
# my_batch_size=200

# #Train the model
# my_model = build_model(learning_rate)
# trained_weight, trained_bias, epochs, rmse = train_model(my_model, df,
#                                                          ytrain, epochs,
#                                                          my_batch_size)

# tf.saved_model.save(my_model, 'my_model')

# tfjs.converters.convert_tf_saved_model('my_model', 'tfjs_model')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

#Load the training file and label
df = pd.read_csv('Cleaned_Electricity_Data.csv')
df = df.drop(columns = ['index', 'Date','Time'])
ytrain = df.drop(columns=['Voltage', 'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
df = df.drop(columns = ['Global_reactive_power','Global_active_power'])

#Load the validation file and label
dfeval = pd.read_csv('PredictingValue.csv')
dfeval = dfeval.drop(columns = ['index', 'Date','Time'])
yeval = dfeval.drop(columns=['Voltage', 'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
dfeval = dfeval.drop(columns = ['Global_reactive_power','Global_active_power'])

#build the Sequential model with 2 hidden layers. The first layer has 5 nodes, 2 hidden layers have 10 nodes and the output layer has 2 nodes
def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10,
                                  input_shape=(5,)))
    model.add(tf.keras.layers.Dense(units = 10))
    model.add(tf.keras.layers.Dense(units=2))
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
   
    return model

def train_model(model, feature, label, epochs, batch_size):
    history = model.fit(x=feature,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse

learning_rate=0.005
epochs=20
my_batch_size=200

#Train the model
my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, df,
                                                         ytrain, epochs,
                                                         my_batch_size)

# Save the TensorFlow model
my_model.save("my_model2")
# Convert and save the model in TensorFlow.js format
tfjs.converters.save_keras_model(my_model, "tfjs_model2")