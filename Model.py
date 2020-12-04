import tensorflow as tf

def linear(inputs, output_size, name_scope):
	shape = [inputs.get_shape()[1], output_size]
	with tf.variable_scope(name_scope):
		w = tf.get_variable(
				"w",
				[inputs.get_shape()[1], output_size],
				initializer = tf.random_normal_initializer(stddev=0.1)
			)
		b = tf.get_variable(
				"b",
				[output_size],
				initializer = tf.constant_initializer(0.0)
			)
	return tf.matmul(inputs, w) + b

def model(inputs):
	f1 = linear(
			inputs = inputs, 
			output_size = 256, 
			name_scope = "f1"
		)
	f2 = linear(
			inputs = tf.nn.relu(f1), 
			output_size = 128, 
			name_scope = "f2"
		)
	f3 = linear(inputs = tf.nn.relu(f2), 
			output_size = 64, 
			name_scope = "f3"
		)
	f4 = linear(inputs = tf.nn.relu(f3), 
			output_size = 32, 
			name_scope = "f4"
		)
	f5 = linear(inputs = tf.nn.relu(f4), 
			output_size = 16, 
			name_scope = "f5"
		)
	f6 = linear(inputs = tf.nn.relu(f5), 
			output_size = 32, 
			name_scope = "f6"
		)
	f7 = linear(inputs = tf.nn.relu(f6), 
			output_size = 64, 
			name_scope = "f7"
		)
	f8 = linear(inputs = tf.nn.relu(f7), 
			output_size = 128, 
			name_scope = "f8"
		)
	f9 = linear(inputs = tf.nn.relu(f8), 
			output_size = 220, 
			name_scope = "f9"
		)
	return f9

def optimizer(loss, lr):
	train_step = tf.train.AdamOptimizer(lr).minimize(loss)
	return train_step


class network(object):
	def __init__(self, axis, lr):
		self.x = tf.placeholder(
					tf.float32, 
					[None, axis], 
					name = 'x'
				)
		self.y = tf.placeholder(
					tf.float32,
					[None, 220],
					name = "y"
				)
		with tf.name_scope("network"):
			self.pre = model(self.x)

		self.loss = tf.reduce_mean(
						tf.square(self.pre - self.y), 
						name='loss'
					)
		self.opt = optimizer(self.loss, lr)
        
