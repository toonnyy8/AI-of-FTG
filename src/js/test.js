import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"

console.log(tf.memory());

let g = tf.grad((x) => {
    let a = tf.mul(x, 2)
    a = tf.concat([x, tfex.stopGradient(x)])
    return a
})

g(tf.tensor([1, 2, 3])).print()
console.log(tf.memory())