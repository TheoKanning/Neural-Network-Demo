import tensorflow as tf

training_inputs = [[1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [0, 1, 0]]

training_outputs = [[.9],
                    [.1],
                    [.9],
                    [.6]]

x = tf.placeholder(tf.float32, [None, 3])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([3, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.sigmoid(tf.matmul(x, W) + b)

errors = tf.sub(y_, y)
cost = tf.reduce_sum(tf.square(errors))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

print_weights = tf.Print(W, [W], message="Current Weights: ")
print_bias = tf.Print(b, [b], message="Current Bias: ")

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10001):
    sess.run(train_step, feed_dict={x: training_inputs, y_: training_outputs})

    if i in [0, 50, 100, 150, 200, 500, 1000, 2000, 5000, 10000]:
        train_cost = sess.run(cost, feed_dict={x: training_inputs, y_: training_outputs})
        print("step %d, training cost %g" % (i, train_cost))

        output = sess.run(print_weights)
        output = sess.run(print_bias)



