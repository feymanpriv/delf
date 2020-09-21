# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Export DELG tensorflow inference model.

The exported model can be used to jointly extract local and global features. It
may use an image pyramid for multi-scale processing, and will include receptive
field calculation and keypoint selection for the local feature head.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
BASE =os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, BASE)
print(BASE)
from absl import app
from absl import flags
import tensorflow as tf

from delf.python.training.model import delf_model
from delf.python.training.model import delg_model
from delf.python.training.model import export_model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', '/root/paddlejob/workspace/env_run/yangmin09/models/research/delf/delf/python/training/output2/delg_tf2-ckpt',
                    'Path to saved checkpoint.')
flags.DEFINE_string('export_path', '/root/paddlejob/workspace/env_run/yangmin09/models/research/delf/delf/python/training/output2/a.pb', 'Path where model will be exported.')
flags.DEFINE_boolean('delg_global_features', True,
                     'Whether the model uses a DELG-like global feature head.')
flags.DEFINE_float(
    'delg_gem_power', 3.0,
    'Power for Generalized Mean pooling. Used only if --delg_global_features'
    'is present.')
flags.DEFINE_integer(
    'delg_embedding_layer_dim', 2048,
    'Size of the FC whitening layer (embedding layer). Used only if'
    '--delg_global_features is present.')
flags.DEFINE_boolean(
    'block3_strides', True,
    'Whether to apply strides after block3, used for local feature head.')
flags.DEFINE_float('iou', 1.0,
                   'IOU for non-max suppression used in local feature head.')


class _ExtractModule(tf.Module):
  """Helper module to build and save DELG model."""

  def __init__(self,
               delg_global_features=True,
               delg_gem_power=3.0,
               delg_embedding_layer_dim=2048,
               block3_strides=True,
               iou=1.0):
    """Initialization of DELG model.

    Args:
      delg_global_features: Whether the model uses a DELG-like global feature
        head.
      delg_gem_power: Power for Generalized Mean pooling in the DELG model. Used
        only if 'delg_global_features' is True.
      delg_embedding_layer_dim: Size of the FC whitening layer (embedding
        layer). Used only if 'delg_global_features' is True.
      block3_strides: bool, whether to add strides to the output of block3.
      iou: IOU for non-max suppression.
    """
    self._stride_factor = 2.0 if block3_strides else 1.0
    self._iou = iou

    # Setup the DELG model for extraction.
    if delg_global_features:
      self._model = delg_model.Delg(
          block3_strides=block3_strides,
          name='DELG',
          gem_power=delg_gem_power,
          embedding_layer_dim=delg_embedding_layer_dim)
    else:
      self._model = delf_model.Delf(block3_strides=block3_strides, name='DELF')

  def LoadWeights(self, checkpoint_path):
    self._model.load_weights(checkpoint_path)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image'),
      tf.TensorSpec(shape=[None], dtype=tf.float32, name='input_scales'),
      tf.TensorSpec(shape=(), dtype=tf.int32, name='input_max_feature_num'),
      tf.TensorSpec(shape=(), dtype=tf.float32, name='input_abs_thres'),
      tf.TensorSpec(
          shape=[None], dtype=tf.int32, name='input_global_scales_ind')
  ])
  def ExtractFeatures(self, input_image, input_scales, input_max_feature_num,
                      input_abs_thres, input_global_scales_ind):
    extracted_features = export_model_utils.ExtractLocalAndGlobalFeatures(
        input_image, input_scales, input_max_feature_num, input_abs_thres,
        input_global_scales_ind, self._iou,
        lambda x: self._model.build_call(x, training=False),
        self._stride_factor)

    named_output_tensors = {}
    named_output_tensors['boxes'] = tf.identity(
        extracted_features[0], name='boxes')
    named_output_tensors['features'] = tf.identity(
        extracted_features[1], name='features')
    named_output_tensors['scales'] = tf.identity(
        extracted_features[2], name='scales')
    named_output_tensors['scores'] = tf.identity(
        extracted_features[3], name='scores')
    named_output_tensors['global_descriptors'] = tf.identity(
        extracted_features[4], name='global_descriptors')
    return named_output_tensors


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  export_path = FLAGS.export_path
  if os.path.exists(export_path):
    raise ValueError(f'Export_path {export_path} already exists. Please '
                     'specify a different path or delete the existing one.')

  module = _ExtractModule(FLAGS.delg_global_features, FLAGS.delg_gem_power,
                          FLAGS.delg_embedding_layer_dim, FLAGS.block3_strides,
                          FLAGS.iou)

  # Load the weights.
  checkpoint_path = FLAGS.ckpt_path
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9)


  # Setup checkpoint directory.
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=module)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
  print('Checkpoint loaded from ', checkpoint_path)

  # Save the module
  tf.saved_model.save(module, export_path)


if __name__ == '__main__':
  app.run(main)
