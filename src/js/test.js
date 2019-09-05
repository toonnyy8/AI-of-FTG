import * as tf from "@tensorflow/tfjs"
console.log(tf.memory())

let a = tf.tensor([1, 2, 3, 4, 1, 2, 1, 3], [8, 1, 1, 1, 1, 1, 1, 1])
console.log(a)
a.transpose([0, 2, 1, 3, 4, 5, 6, 7])

console.log(tf.memory())