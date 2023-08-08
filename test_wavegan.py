import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '--logdir',
    type=str,
    default='./logdir',
    help='Where saved checkpoints are stored.'
)

parser.add_argument(
    '--outdir',
    type=str,
    default='./gen_outputs',
    help='Where to save generated outputs.'
)


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

parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='Sample rate used to convert vector to wav file.'
)

args = parser.parse_args()
step = args.step
num_examples = args.num_examples
batch_size = 500
num_batches = num_examples // batch_size

checkpoints_dir = args.logdir                           # './logdir'
infer_dir = os.path.join(checkpoints_dir, 'infer')      # './logdir/infer'
out_dir = args.outdir                                   # './gen_outputs'
sample_rate = args.sample_rate                          # default: 16000


# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph(os.path.join(infer_dir, 'infer.meta'))       # saver is created from loading a metagraph
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, os.path.join(checkpoints_dir, f'model.ckpt-{step}'))        # then, the graph (saver) restores the values saved in model.ckpt-xxx

print(f'Loaded model from saved checkpoint after {step} steps.')

# Generate examples, in batches
for i in range(num_batches):

    # Create random latent vectors z for 1 example
    _z = (np.random.rand(batch_size, 100) * 2.) - 1

    # Synthesize G(z)
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(G_z, {z: _z})

    for j in range(batch_size):
        gen_data = _G_z[j, :, 0]
        write(os.path.join(out_dir, f"{i*batch_size + j}.wav"), sample_rate, gen_data)

print(f'Generated {num_examples} outputs to {out_dir} directory.')
