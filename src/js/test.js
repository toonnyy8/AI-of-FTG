import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"

console.log(tf.memory());

tf.initializers.randomUniform({ maxval: 0.1, minval: 0.1 }).apply([1400, 8]).print()

console.log(tf.memory())