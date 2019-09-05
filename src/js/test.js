import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"

console.log(tf.memory())

tf.tidy(() => {
    return tfex.unstack(tf.tensor([1, 2, 3, 4]))
}).forEach((t) => {
    t.print()
})

console.log(tf.memory())