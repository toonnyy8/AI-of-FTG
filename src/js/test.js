import * as tf from "@tensorflow/tfjs"
console.log(tf.memory())

let a = tf.tensor([1, 2, 3, 4, 1, 2, 3, 4], [1, 4, 2])

let layer = tf.layers.dense({ units: 3 })
tf.tidy(() => {
    console.log(layer.apply(a).shape)
    console.log(layer.getWeights())
})

console.log(tf.memory())