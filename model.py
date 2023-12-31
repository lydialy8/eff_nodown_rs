from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers, GaussianNoise
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

layers = VersionAwareLayers()

def EfficientNet(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation='swish',
    blocks_args='default',
    model_name='efficientnet',
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):

  if blocks_args == 'default':
    blocks_args = DEFAULT_BLOCKS_ARGS

  if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  def round_filters(filters, divisor=depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
      new_filters += divisor
    return int(new_filters)

  def round_repeats(repeats):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))

  # Build stem
  x = img_input
  x = GaussianNoise(0.1)(x)
  x = layers.Normalization(axis=bn_axis)(x)

  x = layers.ZeroPadding2D(
      padding=imagenet_utils.correct_pad(x, 3),
      name='stem_conv_pad')(x)
  x = layers.Conv2D(
      round_filters(32),
      2,
      strides=1,
      padding='valid',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='stem_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
  x = layers.Activation(activation, name='stem_activation')(x)

  # Build blocks
  blocks_args = copy.deepcopy(blocks_args)

  b = 0
  blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
  for (i, args) in enumerate(blocks_args):
    assert args['repeats'] > 0
    # Update block input and output filters based on depth multiplier.
    args['filters_in'] = round_filters(args['filters_in'])
    args['filters_out'] = round_filters(args['filters_out'])

    for j in range(round_repeats(args.pop('repeats'))):
      # The first block needs to take care of stride and filter size increase.
      if j > 0:
        args['strides'] = 1
        args['filters_in'] = args['filters_out']
      x = block(
          x,
          activation,
          drop_connect_rate * b / blocks,
          name='block{}{}_'.format(i + 1, chr(j + 97)),
          **args)
      b += 1

  # Build top
  x = layers.Conv2D(
      round_filters(1280),
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='top_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
  x = layers.Activation(activation, name='top_activation')(x)
  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate, name='top_dropout')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name=model_name)

  # Load weights.
  if weights is not None:
    model.load_weights(weights)
  return model


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  # Expansion phase
  filters = filters_in * expand_ratio
  if expand_ratio != 1:
    x = layers.Conv2D(
        filters,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'expand_conv')(
            inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
    x = layers.Activation(activation, name=name + 'expand_activation')(x)
  else:
    x = inputs

  # Depthwise Convolution
  if strides == 2:
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, kernel_size),
        name=name + 'dwconv_pad')(x)
    conv_pad = 'valid'
  else:
    conv_pad = 'same'
  x = layers.DepthwiseConv2D(
      kernel_size,
      strides=strides,
      padding=conv_pad,
      use_bias=False,
      depthwise_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'dwconv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
  x = layers.Activation(activation, name=name + 'activation')(x)

  # Squeeze and Excitation phase
  if 0 < se_ratio <= 1:
    filters_se = max(1, int(filters_in * se_ratio))
    se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
    se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
    se = layers.Conv2D(
        filters_se,
        1,
        padding='same',
        activation=activation,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_reduce')(
            se)
    se = layers.Conv2D(
        filters,
        1,
        padding='same',
        activation='sigmoid',
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_expand')(se)
    x = layers.multiply([x, se], name=name + 'se_excite')

  # Output phase
  x = layers.Conv2D(
      filters_out,
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'project_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
  if id_skip and strides == 1 and filters_in == filters_out:
    if drop_rate > 0:
      x = layers.Dropout(
          drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
    x = layers.add([x, inputs], name=name + 'add')
  return x


def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
  return EfficientNet(
      1.0,
      1.1,
      240,
      0.2,
      model_name='efficientnetb1',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)



def EfficientNetB4(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
  return EfficientNet(
      1.4,
      1.8,
      380,
      0.4,
      model_name='efficientnetb4',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)

def EfficientNetB5(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
  return EfficientNet(
      1.6,
      2.2,
      456,
      0.4,
      model_name='efficientnetb5',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)

def EfficientNetB6(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
  return EfficientNet(
      1.8,
      2.6,
      528,
      0.5,
      model_name='efficientnetb6',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)

def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
  return EfficientNet(
      1.1,
      1.2,
      260,
      0.3,
      model_name='efficientnetb2',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
  return EfficientNet(
      1.2,
      1.4,
      300,
      0.3,
      model_name='efficientnetb3',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)

def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
  return x