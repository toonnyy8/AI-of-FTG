import * as tf from "@tensorflow/tfjs"


declare function registerFuncs(tf): {
    einsum(subscripts: string, ...operands: tf.Tensor[]): tf.Tensor

    matrixBandPart(input: tf.Tensor, numLower: Number, numUpper: Number): tf.Tensor

    stopGradient(x: tf.Tensor): tf.Tensor

    l2Normalize(x: tf.Tensor, axis?: Number, epsilon?: Number): tf.Tensor

    clipByGlobalNorm(tList: tf.Tensor[], clipNorm: Number): [tf.Tensor[], tf.Scalar]

    transpose(x: tf.Tensor, perm: Number[]): tf.Tensor

    stack(tensors: tf.Tensor[], axis: Number): tf.Tensor

    unstack(x: tf.Tensor, axis: Number): tf.Tensor[]

    tile(x: tf.Tensor, reps: Number[]): tf.Tensor

    softmax(logits: tf.Tensor, dim?: Number): tf.Tensor

    softmaxCrossEntropyWithLogits(logits: tf.Tensor, labels: tf.Tensor, dim?: Number): tf.Tensor
}