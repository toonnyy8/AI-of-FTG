import "@babel/polyfill"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"

console.log(tf.memory())

<<<<<<< HEAD
let a = tf.tensor([1, 2, 3, 4, 5, 6], [1, 2, 3])
a.print()
const time1 = tf.time(() => tf.transpose(a, [2, 0, 1]).print());
time1.then((time) => { console.log(`tf.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

const time2 = tf.time(() => tfex.transpose(a, [2, 0, 1]).print());
time2.then((time) => { console.log(`tfex.transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })
    // const a = tf.tensor([1, 2, 3, 4], [2, 2]);
    // const b = tf.tensor([5, 6, 7, 8], [2, 2]);
    // tf.unstack(a, 1).forEach(t => t.print())
    // tfex.unstack(a, 1).forEach(t => t.print())
=======
let a = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])
const time1 = tf.time(() => tf.transpose(a));
time1.then((time) => { console.log(`transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

const time2 = tf.time(() => tfex.largeRankTranspose(a));
time2.then((time) => { console.log(`largeRankTranspose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })
>>>>>>> 1b88e4c7ad986f4baaa042e711679700182aa868

console.log(tf.memory())