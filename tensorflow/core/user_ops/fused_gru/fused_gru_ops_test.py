from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops import gru_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.client import timeline

from tensorflow.python.platform import test

from plugins.fused_gru import fused_gru_ops


class GRUBlockFusedCellTest(test.TestCase):
    def testGRUBlockFusedCellMultiStep(self):
        with self.test_session(use_gpu=True, graph=ops.Graph()) as sess:
            batch_size = 256
            cell_size = 256
            input_size = 256
            time_steps = 16

            # Random initializers.
            initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=19890212)

            # Inputs
            inputs = []
            for _ in range(time_steps):
                inp = ops.convert_to_tensor(
                    np.random.randn(batch_size, input_size), dtype=dtypes.float32)
                inputs.append(inp)
            stacked_inputs = array_ops.stack(inputs)

            # Output from the basic GRU cell implementation.
            with vs.variable_scope("test", initializer=initializer):
                w_ru = vs.get_variable(
                    "fused_gru_cell/gates/kernel",
                    shape=[input_size + cell_size, cell_size * 2],
                    dtype=dtypes.float32)
                w_c = vs.get_variable(
                    "fused_gru_cell/candidate/kernel",
                    shape=[input_size + cell_size, cell_size],
                    dtype=dtypes.float32)
                b_ru = vs.get_variable(
                    "fused_gru_cell/gates/bias", shape=[cell_size * 2],
                    dtype=dtypes.float32,
                    initializer=init_ops.zeros_initializer())
                b_c = vs.get_variable(
                    "fused_gru_cell/candidate/bias", shape=[cell_size],
                    dtype=dtypes.float32,
                    initializer=init_ops.zeros_initializer())

                w = vs.get_variable(
                    "rnn/lstm_cell/kernel",
                    shape=[input_size + cell_size, cell_size * 4],
                    dtype=dtypes.float32)
                b = vs.get_variable(
                    "rnn/lstm_cell/bias",
                    shape=[cell_size * 4],
                    dtype=dtypes.float32,
                    initializer=init_ops.zeros_initializer())
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                lstm_cell = lstm_ops.LSTMBlockFusedCell(cell_size, cell_clip=0,
                                                        reuse=True, name="rnn/lstm_cell")
                lstm_outputs_op, _ = lstm_cell(
                    stacked_inputs, dtype=dtypes.float32)
                sess.run([variables.global_variables_initializer()])
                xs = [w, b]
                for _ in range(10):
                    sess.run(lstm_outputs_op)
                t0 = time.time()
                lstm_outputs = sess.run(lstm_outputs_op,
                                        options=options, run_metadata=run_metadata)
                t1 = time.time()
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('lstm_decode.json', 'w') as f:
                    f.write(chrome_trace)

                for _ in range(10):
                    sess.run(gradients_impl.gradients(lstm_outputs_op, inputs))
                t2 = time.time()
                lstm_grad = sess.run(gradients_impl.gradients(lstm_outputs_op, inputs),
                                     options=options, run_metadata=run_metadata)
                t3 = time.time()
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('lstm_grad.json', 'w') as f:
                    f.write(chrome_trace)

                for _ in range(10):
                    sess.run(gradients_impl.gradients(lstm_outputs_op, xs))
                t4 = time.time()
                lstm_wgrad = sess.run(gradients_impl.gradients(lstm_outputs_op, xs),
                                      options=options, run_metadata=run_metadata)
                t5 = time.time()
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('lstm_wgrad.json', 'w') as f:
                    f.write(chrome_trace)

                gru_cell = fused_gru_ops.GRUBlockFusedCell(cell_size, reuse=True)
                gru_outputs_op, _ = gru_cell(
                    stacked_inputs, dtype=dtypes.float32)
                sess.run([variables.global_variables_initializer()])
                xxs = [w_ru, w_c, b_ru, b_c]

                for _ in range(10):
                    sess.run(gru_outputs_op)
                t6 = time.time()
                gru_outputs = sess.run(gru_outputs_op,
                                       options=options, run_metadata=run_metadata)
                t7 = time.time()
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('gru_decode.json', 'w') as f:
                    f.write(chrome_trace)

                for _ in range(10):
                    sess.run(gradients_impl.gradients(gru_outputs_op, inputs))
                t8 = time.time()
                gru_grad = sess.run(gradients_impl.gradients(gru_outputs_op, inputs),
                                    options=options, run_metadata=run_metadata)
                t9 = time.time()
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('gru_grad.json', 'w') as f:
                    f.write(chrome_trace)

                for _ in range(10):
                    sess.run(gradients_impl.gradients(gru_outputs_op, xxs))
                t10 = time.time()
                gru_wgrad = sess.run(gradients_impl.gradients(gru_outputs_op, xxs),
                                     options=options, run_metadata=run_metadata)
                t11 = time.time()
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('gru_wgrad.json', 'w') as f:
                    f.write(chrome_trace)

            print('-------LSTM-------')
            print('lstm decode', t1 - t0)
            print('lstm grad', t3 - t2)
            print('lstm wgrad', t5 - t4)
            print('')
            print('-------GRU-------')
            print('gru decode', t7 - t6)
            print('gru grad', t9 - t8)
            print('gru wgrad', t11 - t10)
            print('')
            print('----Difference----')
            print('decode', (t7 - t6) - (t1 - t0))
            print('grad', (t9 - t8) - (t3 - t2))
            print('wgrad', (t11 - t10) - (t5 - t4))
            self.assertEqual(len(gru_outputs), len(lstm_outputs))
            self.assertEqual(len(gru_grad), len(lstm_grad))
            self.assertEqual(len(gru_wgrad), len(lstm_wgrad))


if __name__ == "__main__":
    test.main()
