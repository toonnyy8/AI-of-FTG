import * as tf from "@tensorflow/tfjs"

declare function einsum(subscripts : string, ...operands:tf.Tensor[]):tf.Tensor