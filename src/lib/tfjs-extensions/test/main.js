import "@babel/polyfill"

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


const time = tf.time(() => tfex.tool.sequenceTidy((x, y) => {
    return x.mul(y)
}).next((x) => {
    console.log(x)
    return x.unstack()
}).run(tf.tensor([1, 2, 3]), tf.tensor([4, 5, 6])));

time.then((time) => { console.log(`tfex.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

console.log(tf.memory())