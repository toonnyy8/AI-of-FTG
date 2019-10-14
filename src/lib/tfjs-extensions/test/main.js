import "core-js/stable"
import "regenerator-runtime/runtime"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"

console.log(tf.memory())

// let a = tf.tensor([1, 2, 3, 4, 5, 6], [1, 2, 3])
// a.print()
// const time1 = tf.time(() => tf.transpose(a, [2, 0, 1]).print());
// time1.then((time) => { console.log(`tf.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

// const time2 = tf.time(() => tfex.transpose(a, [2, 0, 1]).print());
// time2.then((time) => { console.log(`tfex.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })
//     // const a = tf.tensor([1, 2, 3, 4], [2, 2]);
//     // const b = tf.tensor([5, 6, 7, 8], [2, 2]);
//     // tf.unstack(a, 1).forEach(t => t.print())
//     // tfex.unstack(a, 1).forEach(t => t.print())


// const time = tf.time(() => tfex.tool.sequenceTidy((x, y) => {
//     return x.mul(y)
// }).next((x) => {
//     console.log(x)
//     return x.unstack()
// }).run(tf.tensor([1, 2, 3]), tf.tensor([4, 5, 6])));

// time.then((time) => { console.log(`tfex.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

// let a = tfex.tool.tensorPtr(tf.tensor([1, 2, 3]))
// a.assign(tf.mul(a.read(), a.read()))
// a.assign(tf.variable(a.read()))

// let l = tfex.tool.tensorPtrList([tf.tensor([1, 2])])
// l.read(0).print()
// l.reName(0, "tB").read("tB").print()
// l.assign({ "tB": tf.tensor([5, 6, 7, 8, 9, 10, 50, 81, 7, 69]) }).read("tB").print()

// l.sequence((tptrList) => {
//     tptrList.assign({ "tA": tf.tensor([6]) })
// }).sequence(tptrList => {
//     tptrList.assign({ "tB": tptrList.read("tB").mul(tptrList.read("tA")) })
// }).read("tB").print()
// l.assign({ tA: null })

// let a = tf.range(0, 24 * 24 * 8, 1).reshape([24, 24, 8])
// let b = tf.range(0, 24 * 24 * 8, 1).reshape([24, 8, 24])
// const time = tf.time(() => console.log(tfex.einsum("ijk,jkl->il", a, b)))
// time.then((time) => { console.log(`tfex.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

// const time = tf.time(() => tfex.tile(tf.tensor([1, 2, 3, 4], [2, 2]), [1, 2]).print())
// time.then((time) => { console.log(`tfex.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })
// let x = tf.tensor([0, 0, 17])
// let y = tf.tensor([0, 0, 1])
// tfex.softmax(x).print()
// tf.softmax(x).print()
// tfex.softmaxCrossEntropyWithLogits(x, y).print()

// tfex.scope.variableScope("1").variableScope("2").variableScope("3").variableScope("4").getVariable("a", [1])
// let scopeSave = tfex.scope.save()
// console.log(scopeSave)
// let save = tfex.sl.save(scopeSave)
// console.log(save)
// let load = tfex.sl.load(save)
// console.log(load)
// console.log(tfex.scope.variableScope("test").load(load))

let subLayer = tfex.layers.lambda({ func: (a, b) => { return tf.sub(a, b) }, outputShape: [1] })
console.log(subLayer.apply([tf.input({ shape: [1] }), tf.input({ shape: [1] })]))
subLayer.apply([tf.tensor([1]), tf.tensor([10])]).print()

console.log(tf.memory())