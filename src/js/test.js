import "core-js/stable"
import "regenerator-runtime/runtime"
import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"

console.log(tf.memory());

let a = tfex.scope.variableScope("a")
a.getVariable("test", [2, 2])
a.variableScope("b").getVariable("test", [20, 10])
console.log(a.save())

console.log(tf.memory())