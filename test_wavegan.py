import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
import argparse
import os

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, 'model.ckpt')

# Create random latent vectors z for 1 example
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

out_dir = './gen_outputs'
sample_rate = 16000
for i in range(50):
    gen_data = _G_z[i, :, 0]
    write(os.path.join(out_dir, f"{i}.wav"), sample_rate, gen_data)