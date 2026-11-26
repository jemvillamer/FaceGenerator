"""CPU compatibility patches for the pretrained StyleGAN v1 model.

The pretrained pickle builds its graph with NCHW ops, which TensorFlow 1.15 does
not support on CPU for several ops. install() wraps the affected tf.nn functions
so NCHW calls are transposed to NHWC around the op.

Call install() before the model is unpickled (that is when the graph is built).
model.py does this on import. Calling it more than once is harmless.
"""
import tensorflow as tf

_PATCH_MARK = '_facegen_cpu_patch'


def _cpu_conv2d_transpose(orig):
    def conv2d_transpose(value, filters=None, output_shape=None, strides=None,
                         padding='SAME', data_format='NHWC', dilations=None,
                         name=None, filter=None):
        filters = filters if filters is not None else filter
        if data_format == 'NCHW':
            value = tf.transpose(value, [0, 2, 3, 1])
            if isinstance(output_shape, (list, tuple)):
                output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
            else:
                output_shape = tf.stack([output_shape[0], output_shape[2], output_shape[3], output_shape[1]])
            if strides is not None and len(strides) == 4:
                strides = [strides[0], strides[2], strides[3], strides[1]]
            result = orig(value, filters, output_shape, strides,
                          padding, 'NHWC', dilations, name)
            return tf.transpose(result, [0, 3, 1, 2])
        return orig(value, filters, output_shape, strides,
                    padding, data_format, dilations, name)
    return conv2d_transpose


def _cpu_depthwise_conv2d(orig):
    def depthwise_conv2d(input, filter, strides, padding, rate=None,
                         name=None, data_format=None):
        if data_format == 'NCHW':
            input = tf.transpose(input, [0, 2, 3, 1])
            strides = [strides[0], strides[2], strides[3], strides[1]]
            result = orig(input, filter, strides, padding,
                          rate=rate, name=name, data_format='NHWC')
            return tf.transpose(result, [0, 3, 1, 2])
        return orig(input, filter, strides, padding,
                    rate=rate, name=name, data_format=data_format)
    return depthwise_conv2d


def _cpu_avg_pool(orig):
    def avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None, input=None):
        val = value if value is not None else input
        if data_format == 'NCHW':
            val = tf.transpose(val, [0, 2, 3, 1])
            ksize   = [ksize[0],   ksize[2],   ksize[3],   ksize[1]]
            strides = [strides[0], strides[2], strides[3], strides[1]]
            result = orig(val, ksize, strides, padding, 'NHWC', name)
            return tf.transpose(result, [0, 3, 1, 2])
        return orig(val, ksize, strides, padding, data_format, name)
    return avg_pool


def _cpu_conv2d(orig):
    def conv2d(input, filter=None, strides=None, padding=None, use_cudnn_on_gpu=True,
               data_format='NHWC', dilations=None, name=None, filters=None):
        filt = filter if filter is not None else filters
        if data_format == 'NCHW':
            input = tf.transpose(input, [0, 2, 3, 1])
            if strides is not None and len(strides) == 4:
                strides = [strides[0], strides[2], strides[3], strides[1]]
            result = orig(input, filt, strides, padding,
                          use_cudnn_on_gpu, 'NHWC', dilations, name)
            return tf.transpose(result, [0, 3, 1, 2])
        return orig(input, filt, strides, padding,
                    use_cudnn_on_gpu, data_format, dilations, name)
    return conv2d


_PATCHES = {
    'conv2d_transpose': _cpu_conv2d_transpose,
    'depthwise_conv2d': _cpu_depthwise_conv2d,
    'avg_pool': _cpu_avg_pool,
    'conv2d': _cpu_conv2d,
}


def install():
    """Patch tf.nn for CPU. Functions already patched are left alone."""
    for name, make_patch in _PATCHES.items():
        orig = getattr(tf.nn, name)
        if getattr(orig, _PATCH_MARK, False):
            continue
        patched = make_patch(orig)
        setattr(patched, _PATCH_MARK, True)
        setattr(tf.nn, name, patched)
