from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from lstm_model import _LSTMModel
from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model

reader = tf.contrib.timeseries.CSVReader('close.csv')
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=20, window_size=40)
#
# data = pd.read_csv('gold.csv', index_col='Date')['SLV'].dropna()
# data = data.reset_index(drop=True)
# data.to_csv('close.csv')


with tf.Session() as sess:
     batch_data = train_input_fn.create_batch()
     data = reader.read_full()
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
     #print(sess.run(data))
     one_batch = sess.run(batch_data[0])
     coord.request_stop()

# print(one_batch)
#
# ar = tf.contrib.timeseries.ARRegressor(
#     periodicities=5,
#     input_window_size=30,
#     output_window_size=10,
#     num_features=1,
#     loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS
# )
# ar.train(input_fn=train_input_fn, steps=60000)
# evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
# evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
# (predictions,) = tuple(ar.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
#     evaluation, steps=3100
# )))


#print(np.array(data['times']))
#print(np.array(evaluation['mean']).shape)
# plt.figure(figsize=(15, 5))
# plt.plot(range(3150), np.array(evaluation['mean']).reshape(-1), label='eva')
# plt.plot(range(3100), np.array(predictions['mean']).reshape(-1), label='pre')
# plt.show()

estimator = ts_estimators.TimeSeriesRegressor(
    model=_LSTMModel(num_features=1, num_units=128),
    optimizer=tf.train.AdamOptimizer()
)
estimator.train(input_fn=train_input_fn, steps=200000)
evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
(predictions,) = tuple(estimator.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
    evaluation, steps=500
)))
observed_times = evaluation["times"][0]
observed = evaluation["observed"][0, :, :]
evaluated_times = evaluation["times"][0]
evaluated = evaluation["mean"][0]
predicted_times = predictions['times']
predicted = predictions["mean"]

plt.figure(figsize=(15, 5))
plt.axvline(3721, linestyle="dotted", linewidth=4, color='r')
observed_lines = plt.plot(observed_times, observed, label="observation", color="k")
evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
predicted_lines = plt.plot(predicted_times, predicted, label="prediction", color="r")
plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],
             loc="upper left")
plt.show()

