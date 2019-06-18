
		self.CM_0 = Conv2D(filters=message_size * 2   , kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer, name='CM_0') # * 2







	def edge2node(self, edge, size=message_size, name=None): #
		hor0 = tf.slice(edge, [0, 0, 0,    0], [-1, input_dimensions, input_dimensions_m1, size])
		hor1 = tf.slice(edge, [0, 0, 0, size], [-1, input_dimensions, input_dimensions_m1, size])
		ver0 = tf.slice(edge, [0, input_dimensions, 0,    0], [-1, input_dimensions, input_dimensions_m1, size])
		ver1 = tf.slice(edge, [0, input_dimensions, 0, size], [-1, input_dimensions, input_dimensions_m1, size])
		hor_ = tf.pad(hor0, self.pad0) + tf.pad(hor1, self.pad1)
		ver_ = tf.pad(ver0, self.pad0) + tf.pad(ver1, self.pad1)
		return tf.add(hor_, tf.transpose(ver_, [0, 2, 1, 3]), name=name)







repar_degree = main_layers.edge2node(tf.ones([1, input_dimensions_2, input_dimensions_m1, 2]), size=1, name='repar_degree') # 2