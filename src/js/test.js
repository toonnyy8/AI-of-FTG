import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"

console.log(tf.memory())

tf.tidy(() => {
    tfex.layers.layerNormalization({ axis: 0 }).apply(tf.tensor([1, 2, 3]))
})

console.log(tf.memory())