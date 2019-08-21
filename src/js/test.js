import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim

// import * as transformerXL from "./MirageNet/transformerXL"
// console.log(transformerXL)
console.log(tf.memory())

let input = tf.tensor([[0, 1, 2, 3],
[-1, 0, 1, 2],
[-2, -1, 0, 1],
[-3, -2, -1, 0]])

tfex.matrixBandPart(input, 0, -1).print()

console.log(tf.memory())