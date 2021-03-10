import random
import math
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')


def do_generate_x_y(isTrain, batch_size, seqlen):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1
        xlim_start = seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand
        xlim_end = seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand
        sin_data = amp_rand * np.sin(np.linspace(xlim_start, xlim_end, seqlen * 2))
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 1.5 + 0.5
        amp_rand = random.random() * 1.2
        xlim_start = seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand
        xlim_end = seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand
        sig_data = amp_rand * np.cos(np.linspace(xlim_start, xlim_end, seqlen * 2)) + sin_data
        batch_x.append(np.array([sig_data[:seqlen]]).T)
        batch_y.append(np.array([sig_data[seqlen:]]).T)
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y


def generate_data(isTrain, batch_size):
    seq_length = 15
    if isTrain:
        return do_generate_x_y(isTrain, batch_size, seqlen=seq_length)
    else:
        return do_generate_x_y(isTrain, batch_size, seqlen=seq_length*2)


sample_now, sample_f = generate_data(isTrain=True, batch_size=10)
seq_length = sample_now.shape[0]
batch_size = 10
output_dim = input_dim = sample_now.shape[-1]
hidden_dim = 12
layers_stacked_count = 2

learning_rate = 0.04
nb_iters = 100
l2_reg = 0.003

tf.reset_default_graph()

encoder_input = []
expected_output = []
decode_input = []

for i in range(seq_length):
    encoder_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))
    expected_output.append(tf.placeholder(tf.float32, shape=(None, output_dim)))
    decode_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))

tcells = []
for i in range(layers_stacked_count):
    tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))
Mcell = tf.contrib.rnn.MultiRNNCell(tcells)

dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_input,decode_input,Mcell)

reshaped_outputs = []
for i in dec_outputs:
    reshaped_outputs.append(tf.contrib.layers.fully_connected(i, output_dim, activation_fn=None))

output_loss = 0
for _y, _Y in zip(reshaped_outputs, expected_output):
    output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

reg_loss = 0
for tf_var in tf.trainable_variables():
    if not ('fully_connected' in tf_var.name):
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

loss = output_loss + l2_reg * reg_loss
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()


def train_batch(batch_size):
    X, Y = generate_data(isTrain=True, batch_size=batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(Y[0])], Y[:-1]), axis=0)
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size):
    X, Y = generate_data(isTrain=True, batch_size=batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(Y[0])], Y[:-1]), axis=0)
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    output_lossv,reg_lossv,loss_t = sess.run([output_loss, reg_loss, loss], feed_dict)
    print("----------------------------")
    print(output_lossv, reg_lossv)
    return loss_t


train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    if t % 50 == 0:
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, nb_iters, train_loss, test_loss))
print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))

plt.figure(figsize=(12, 6))
plt.plot(np.array(range(0, len(test_losses))) / float(len(test_losses) - 1) * (len(train_losses) - 1),
         np.log(test_losses), label="Test loss")
plt.plot(np.log(train_losses), label = "Test loss")
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()

nb_predictions = 1
print("visualise {} predictions data:".format(nb_predictions))

preout = []
X, Y = generate_data(isTrain=False, batch_size=nb_predictions)
print(np.shape(X), np.shape(Y))
for tt in range(seq_length):
    feed_dict = {encoder_input[t]: X[t+tt] for t in range(seq_length)}
    feed_dict.update({expected_output[t]: Y[t+tt] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(Y[0])], Y[tt:seq_length+tt-1]), axis=0)
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    outputs = sess.run([reshaped_outputs], feed_dict)
    outputs = outputs[0]
    preout.append(outputs[-1])

print(np.shape(preout))
preout = np.reshape(preout, [seq_length, nb_predictions, output_dim])

for j in range(nb_predictions):
    plt.figure(figsize=(12,3))
    for k in range(output_dim):
        past = X[:, j, k]
        expected = Y[seq_length-1:, j, k]
        pred = preout[:, j, k]
        label1 = "past" if k == 0 else "_nolegend_"
        label2 = "future" if k == 0 else "_nolegend_"
        label3 = "Pred" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected) + len(past)), expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)
    plt.legend(loc='best')
    plt.title("Predictions vs. future")
    plt.show()
































