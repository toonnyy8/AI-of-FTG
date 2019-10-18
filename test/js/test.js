import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)
import * as fs from "fs"
let weight = fs.readFileSync(__dirname + "/../../src/param/w.bin")

import * as cnnNLP from "../../src/js/MirageNet/cnnNLP"
console.log(tf.memory());

const input1 = tf.input({ shape: [10] });
const input2 = tf.input({ shape: [20] });
const dense1 = tf.layers.dense({ units: 4 }).apply(input1);
const dense2 = tf.layers.dense({ units: 8 }).apply(input2);
const concat = tf.layers.concatenate().apply([dense1, dense2]);
const output1 =
    tf.layers.dense({ units: 3, activation: 'softmax' }).apply(concat);
const output2 =
    tf.layers.dense({ units: 3, activation: 'softmax' }).apply(concat);
const model = tf.model({ inputs: [input1, input2], outputs: [output1, output2] });
model.summary();

model.predict([tf.ones([1, 10]), tf.ones([1, 20])])[0].print()

console.log(tf.memory())