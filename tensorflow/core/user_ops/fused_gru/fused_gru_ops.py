from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os.path as osp
import tensorflow as tf
import numpy

from tensorflow.python.framework import ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import resource_loader

#compiler = OperaterCompiler('fused_gru', osp.dirname(osp.abspath(__file__)), ['/usr/local'])
#compiler.record_cpu_basis(
#    ['fused_gru_ops.cc', 'fused_gru_ops_reg.cc', 'blas_gemm.cc'],
#    '_fused_gru_ops.so'
#)
#compiler.record_gpu_kernel_builders(
#    ['fused_gru_ops.cu.cc']
#)
#
#_fused_gru_ops_so = compiler.compile()

_fused_gru_ops_so = tf.load_op_library('./_fused_gru_ops.so')

@ops.RegisterGradient("BlockGRU")
def _BlockGRUGrad(op, *grad):
    """Gradient for BlockGRU."""
    x, h_prev, w_ru, w_c, b_ru, b_c = op.inputs
    r, u, c, h = op.outputs
    _, _, _, d_h = grad

    x_grad, h_prev_grad, w_ru_grad, w_c_grad, b_ru_grad, b_c_grad = _fused_gru_ops_so.block_gru_grad(
        x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, h, d_h, seq_len_max=op.get_attr("seq_len_max"))
    return [x_grad, h_prev_grad, w_ru_grad, w_c_grad, b_ru_grad, b_c_grad]


class GRUBlockWrapper(base_layer.Layer):
    """This is a helper class that provides housekeeping for LSTM cells.
    This may be useful for alternative LSTM and similar type of cells.
    The subclasses must implement `_call_cell` method and `num_units` property.
    """

    @abc.abstractproperty
    def cell_size(self):
        """Number of units in this cell (output dimension)."""
        pass

    @abc.abstractmethod
    def _call_cell(self, inputs, initial_output, dtype,
                   sequence_length):
        """Run this LSTM on inputs, starting from the given state.
        This method must be implemented by subclasses and does the actual work
        of calling the cell.
        Args:
          inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
          initial_cell_state: initial value for cell state, shape `[batch_size,
            self._num_units]`
          initial_output: initial value of cell output, shape `[batch_size,
            self._num_units]`
          dtype: The data type for the initial state and expected output.
          sequence_length: Specifies the length of each sequence in inputs. An int32
            or int64 vector (tensor) size [batch_size], values in [0, time_len) or
              None.
        Returns:
          A pair containing:
          - State: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
          - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
        """
        pass

    def call(self, inputs, initial_output=None, dtype=None, sequence_length=None):
        """Run this LSTM on inputs, starting from the given state.
        Args:
          inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
          initial_output: a tuple `(initial_cell_state, initial_output)` with tensors
            of shape `[batch_size, self._num_units]`. If this is not provided, the
            cell is expected to create a zero initial state of type `dtype`.
          dtype: The data type for the initial state and expected output. Required
            if `initial_state` is not provided or RNN state has a heterogeneous
            dtype.
          sequence_length: Specifies the length of each sequence in inputs. An
            `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
            time_len).`
            Defaults to `time_len` for each element.
        Returns:
          A pair containing:
          - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
            or a list of time_len tensors of shape `[batch_size, output_size]`,
            to match the type of the `inputs`.
          - Final state: a tuple `(cell_state, output)` matching `initial_state`.
        Raises:
          ValueError: in case of shape mismatches
        """
        is_list = isinstance(inputs, list)
        if is_list:
            inputs = array_ops.stack(inputs)
        inputs_shape = inputs.get_shape().with_rank(3)
        if not inputs_shape[2]:
            raise ValueError("Expecting inputs_shape[2] to be set: %s" % inputs_shape)
        batch_size = inputs_shape[1].value
        if batch_size is None:
            batch_size = array_ops.shape(inputs)[1]
        time_len = inputs_shape[0].value
        if time_len is None:
            time_len = array_ops.shape(inputs)[0]

        # Provide default values for initial_state and dtype
        if initial_output is None:
            if dtype is None:
                raise ValueError("Either initial_state or dtype needs to be specified")
            z = array_ops.zeros(
                array_ops.stack([batch_size, self.cell_size]), dtype=dtype)
            initial_output = z
        else:
            output_shape = initial_output.get_shape().with_rank(2)
            if output_shape[0] != batch_size or output_shape[1] != self.cell_size:
                raise ValueError("Shape of initial_output should be [batch_size, cell_size]")
            if dtype is None:
                dtype = initial_output[0].dtype

        # create the actual cell
        if sequence_length is not None:
           sequence_length = ops.convert_to_tensor(sequence_length)
        outputs = self._call_cell(inputs, initial_output, dtype, sequence_length)

        if sequence_length is not None:
            # Mask out the part beyond sequence_length
            mask = array_ops.transpose(
                array_ops.sequence_mask(sequence_length, time_len, dtype=dtype),
                [1, 0])
            mask = array_ops.tile(
                array_ops.expand_dims(mask, [-1]), [1, 1, self.cell_size])
            outputs *= mask
            # Prepend initial states to cell_states and outputs for indexing to work
            # correctly,since we want to access the last valid state at
            # sequence_length - 1, which can even be -1, corresponding to the
            # initial state.
            mod_outputs = array_ops.concat(
                [array_ops.expand_dims(initial_output, [0]), outputs], 0)
            final_output = self._gather_states(mod_outputs, sequence_length,
                                               batch_size)
        else:
            # No sequence_lengths used: final state is the last state
            final_output = outputs[-1]

        if is_list:
            # Input was a list, so return a list
            outputs = array_ops.unstack(outputs)

        # final_output = rnn_cell_impl.LSTMStateTuple(final_output, final_output)
        return outputs, final_output

    def _gather_states(self, data, indices, batch_size):
        """Produce `out`, s.t. out(i, j) = data(indices(i), i, j)."""
        return array_ops.gather_nd(
            data, array_ops.stack([indices, math_ops.range(batch_size)], axis=1))


class GRUBlockFusedCell(GRUBlockWrapper):
    """FusedRNNCell implementation of LSTM.
    This is an extremely efficient LSTM implementation, that uses a single TF op
    for the entire LSTM. It should be both faster and more memory-efficient than
    LSTMBlockCell defined above.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    The variable naming is consistent with `rnn_cell_impl.LSTMCell`.
    """

    def __init__(self,
                 cell_size,
                 reuse=None,
                 name="fused_gru_cell"):
        """Initialize the LSTM cell.
        Args:
          cell_size: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          cell_clip: clip the cell to this value. Default is no cell clipping.
          use_peephole: Whether to use peephole connections or not.
          reuse: (optional) boolean describing whether to reuse variables in an
            existing scope.  If not `True`, and the existing scope already has the
            given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.  By default this is "lstm_cell", for variable-name compatibility
            with `tf.nn.rnn_cell.LSTMCell`.
        """
        super(GRUBlockFusedCell, self).__init__(_reuse=reuse, name=name)
        self._cell_size = cell_size

        # Inputs must be 3-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=3)

    @property
    def cell_size(self):
        """Number of units in this cell (output dimension)."""
        return self._cell_size

    def build(self, input_shape):
        input_size = input_shape[2].value
        if input_size is None:
            raise ValueError("Expecting input_size to be set.")

        self._gate_kernel = self.add_variable(
            "gates/kernel", [input_size + self._cell_size, self._cell_size * 2])
        self._gate_bias = self.add_variable(
            "gates/bias", [self._cell_size * 2],
            initializer=init_ops.constant_initializer(1.0))
        self._candidate_kernel = self.add_variable(
            "candidate/kernel", [input_size + self._cell_size, self._cell_size])
        self._candidate_bias = self.add_variable(
            "candidate/bias", [self._cell_size],
            initializer=init_ops.constant_initializer(0.0))

        self.built = True

    def _call_cell(self,
                   inputs,
                   initial_output=None,
                   dtype=None,
                   sequence_length=None):
        """Run this LSTM on inputs, starting from the given state.
        Args:
          inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
          initial_cell_state: initial value for cell state, shape `[batch_size,
            self._num_units]`
          initial_output: initial value of cell output, shape `[batch_size,
            self._num_units]`
          dtype: The data type for the initial state and expected output.
          sequence_length: Specifies the length of each sequence in inputs. An
            `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
            time_len)` or None.
        Returns:
          A pair containing:
          - Cell state (cs): A `3-D` tensor of shape `[time_len, batch_size,
                             output_size]`
          - Output (h): A `3-D` tensor of shape `[time_len, batch_size,
                        output_size]`
        """

        inputs_shape = inputs.get_shape().with_rank(3)
        time_len = inputs_shape[0].value
        if time_len is None:
            time_len = array_ops.shape(inputs)[0]

        if sequence_length is None:
            max_seq_len = time_len
        else:
            max_seq_len = math_ops.to_int64(math_ops.reduce_max(sequence_length))[0]

        _, _, _, h = _fused_gru_ops_so.block_gru(
            x=inputs,
            h_prev=initial_output,
            w_ru=self._gate_kernel,
            w_c=self._candidate_kernel,
            b_ru=self._gate_bias,
            b_c=self._candidate_bias,
            seq_len_max=max_seq_len)
        return h
