import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../src/lib/tfjs-extensions/src"
import * as fs from "fs"
let weight = fs.readFileSync(__dirname + "/../../src/param/w.bin")

import * as cnnNLP from "../../src/js/MirageNet/cnnNLP"
console.log(tf.memory());
// console.log(tfex.sl.load(weight))
// tfex.scope.variableScope("transformer_XL").load(tfex.sl.load(weight))
// console.log(tfex.scope)

// let a = tfex.scope.variableScope("a")
// a.getVariable("test", [2, 2])
// a.variableScope("b").getVariable("test", [20, 10])
// console.log(a.save())
let model = cnnNLP.buildModel({
    sequenceLen: 60,
    inputNum: 10,
    embInner: [64, 64, 64],
    filters: 64,
    outputInner: [512, 512, 512],
    outputNum: 36
})
let loop = () => {
    tf.tidy(() => {
        tf.argMax(model.predict(tf.randomNormal([1, 60, 10])), 1).print()
    })
    // trainLoop.run()
    requestAnimationFrame(loop)
}
loop()
console.log(tf.memory())