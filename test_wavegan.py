import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--step',
    type=int,
    required=True,
    help='Which checkpoint to load model from.'
)
parser.add_argument(
    '--num_examples',
    type=int,
    required=True,
    help='Number of output wav files to generate.'
)

args = parser.parse_args()
step = args.step
num_examples = args.num_examples

infer_dir = './logdir/infer'
checkpoints_dir = './logdir'
out_dir = './gen_outputs'
sample_rate = 16000


# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph(os.path.join(infer_dir, 'infer.meta'))
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, os.path.join(checkpoints_dir, f'model.ckpt-{step}'))

# Create random latent vectors z for 1 example
_z = (np.random.rand(num_examples, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

for i in range(num_examples):
    gen_data = _G_z[i, :, 0]
    write(os.path.join(out_dir, f"{i}.wav"), sample_rate, gen_data)
