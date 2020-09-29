import * as tf from "@tensorflow/tfjs"

const mish = (x: tf.Tensor) =>
    tf.tidy(() => tf.mul(x, tf.tanh(tf.softplus(x))))



const ae = () => {
    const convs = [tf.layers.separableConv2d({ kernelSize: 3, filters: 64, strides: 2 })]
    const deconvs = [tf.layers.separableConv2d({ kernelSize: 3, filters: 64, strides: 2 })]
    tf.layers.conv2dTranspose
}