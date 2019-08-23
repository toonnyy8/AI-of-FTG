import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim

import * as transformerXL from "./MirageNet/transformerXL"
console.log(transformerXL)
console.log(tf.memory())

let a = tf.initializers.constant({ value: 1 })

console.log(tf.memory())