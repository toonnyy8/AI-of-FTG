import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
import * as fs from "fs"
let weight = fs.readFileSync(__dirname + "/../param/w.bin")
// console.log(tfex.sl.load(weight))
tfex.scope.variableScope("transformer_XL").load(tfex.sl.load(weight))
console.log(tfex.scope)
console.log(tf.memory());

let a = tfex.scope.variableScope("a")
a.getVariable("test", [2, 2])
a.variableScope("b").getVariable("test", [20, 10])
console.log(a.save())

console.log(tf.memory())