import * as tf from "@tensorflow/tfjs"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim

import * as transformerXL from "./MirageNet/transformerXL"
console.log(transformerXL)
console.log(tf.memory())

let a = transformerXL.layers.relMultiheadAttn({
    dModel: 1,
    nHead: 1,
    dHead: 1,
    dropout: 1,
    dropatt: 1,
    isTraining: true,
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
})
a.apply({
    r: tf.tensor([1, 2, 3], [1, 3]), w: tf.tensor([1, 2, 3], [1, 3]), r_wBias: tf.tensor([1, 2, 3], [1, 3]),
    r_rBias: tf.tensor([1, 2, 3], [1, 3]), attnMask: tf.tensor([1, 2, 3], [1, 3]), mems: tf.tensor([1, 2, 3], [1, 3])
})

console.log(tf.memory())
