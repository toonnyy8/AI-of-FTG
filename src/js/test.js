import * as tf from "@tensorflow/tfjs"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim

// import * as transformerXL from "./MirageNet/transformerXL"
// console.log(transformerXL)
console.log(tf.memory())

let a = tf.range(0, 12, 1).reshape([3, 2, 2])
a.expandDims(2).expandDims(3).expandDims(5).print()

console.log(tf.memory())
