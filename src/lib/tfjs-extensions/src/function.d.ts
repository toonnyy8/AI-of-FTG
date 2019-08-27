import * as tf from "@tensorflow/tfjs"

declare function einsum(subscripts: string, ...operands: tf.Tensor[]): tf.Tensor

declare function matrixBandPart(input: tf.Tensor, numLower: Number, numUpper: Number): tf.Tensor

declare function stopGradient(x: tf.Tensor): tf.Tensor

declare function l2Normalize(x: tf.Tensor, axis?: Number, epsilon?: Number): tf.Tensor

declare function clipByGlobalNorm(tList: tf.Tensor[], clipNorm: Number): [tf.Tensor[],tf.Scalar]