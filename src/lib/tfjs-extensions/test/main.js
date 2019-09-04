import "@babel/polyfill"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"

console.log(tf.memory())

let a = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])
const time1 = tf.time(() => tf.transpose(a));
time1.then((time) => { console.log(`transpose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

const time2 = tf.time(() => tfex.largeRankTranspose(a));
time2.then((time) => { console.log(`largeRankTranspose  kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`); })

console.log(tf.memory())