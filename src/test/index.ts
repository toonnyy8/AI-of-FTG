import * as tf from "@tensorflow/tfjs"

const a = tf.fill([60, 20, 50, 16, 10], 1)
const b = tf.fill([60, 20, 50, 8, 10], 1)
console.log(
    tf.matMul(a, b, false, true)
)
tf.linalg.bandPart(
    tf.fill([10, 10], 1),
    -1, 0,
).print()
