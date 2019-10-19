import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)
import * as fs from "fs"
let weight = fs.readFileSync(__dirname + "/../../src/param/w.bin")

import * as cnnNLP from "../../src/js/MirageNet/cnnNLP"
console.log(tf.memory());

{
    let input = tf.input({ shape: [1, 1, 8] })
    let cnnLayer = tf.layers.conv2d({ kernelSize: [1, 1], filters: 64 }).apply(input)
    cnnLayer = tf.layers.conv2d({ kernelSize: [1, 1], filters: 1 }).apply(cnnLayer)
    cnnModel = tf.model({ inputs: [input], outputs: cnnLayer })
    tf.tidy(() => {
        let i = tf.truncatedNormal([1, 1, 1, 8])
        cnnModel.predict(i).print()
    })
}

console.log(tf.memory())

{
    let input = tf.input({ shape: [8] })
    let fcLayer = tf.layers.dense({ units: 64 }).apply(input)
    fcLayer = tf.layers.dense({ units: 1 }).apply(fcLayer)
    fcModel = tf.model({ inputs: [input], outputs: fcLayer })
    tf.tidy(() => {
        let i = tf.truncatedNormal([1, 8])
        fcModel.predict(i).print()
    })
}

console.log(tf.memory())