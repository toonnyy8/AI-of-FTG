import * as tf from "@tensorflow/tfjs"

const canvas = <HTMLCanvasElement>document.getElementById("canvas")

const t = <tf.Tensor3D>tf.fill([50, 50, 1], 1)
const t2 = t.concat([tf.fill([50, 50, 1], 0)], 1).reshape([100, 50, 1])
const t3 = t2.concat([tf.fill([100, 50, 1], 0)], 2).reshape([100, 100, 1])
console.log(t2)

tf.browser.toPixels(<tf.Tensor3D>t3, canvas)