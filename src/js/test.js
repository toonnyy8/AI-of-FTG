import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"

console.log(tf.memory());

tfex.softmax(tf.tensor([[1, 1, 10, 2], [3, 3, 6, 9]]), 0).print()

console.log(tf.memory())