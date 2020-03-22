import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets('MNIST_data', one_hot= True)

n_nodes1 =50
n_chunks = 28
chunk_size=28
rnn_size = 128
n_class = 10
batch_size = 128
hm_epoch = 10

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def Neural_network_model(data):

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_nodes1])),
             'Biases': tf.Variable(tf.random_normal([n_nodes1]))}

    layer1 = {'weights': tf.Variable(tf.random_normal([n_nodes1, n_class])),
                      'Biases': tf.Variable(tf.random_normal([n_class]))}

    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data,[-1, chunk_size])
    data = tf.split(data, n_chunks, axis=0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)

    outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights'])+layer['Biases']

    output1 = tf.matmul(output, layer1['weights']) + layer1['Biases']

    return output1


def Train_neural_network_model(data):

    prediction = Neural_network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epoch):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):

                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape(batch_size, n_chunks ,chunk_size)

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'of ', hm_epoch, 'loss', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('accuracy', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


Train_neural_network_model(x)


# accuracy 0.9746