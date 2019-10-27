import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)
import * as fs from "fs"
let weight = fs.readFileSync(__dirname + "/../w.bin")

console.log(tf.memory());
let w = tfex.sl.load(weight)
Object.keys(w).forEach((key) => {
    console.log(key)
    w[key].print()
})
console.log(tf.memory())