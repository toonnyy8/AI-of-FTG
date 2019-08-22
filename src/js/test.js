import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim

// import * as transformerXL from "./MirageNet/transformerXL"
// console.log(transformerXL)
console.log(tf.memory())

// f(x) = x ^ 2
let h = (x) => {
    return tf.tidy(() => {
        let y = x.square().square()
        return y
    })
}
const f = (x) => {
        return tf.tidy(() => {
            let y = tf.mul(h(x), x)
            return y
        })
    }
    // f'(x) = 2x
const fg = tf.grad(f);

const x = tf.tensor1d([2, 3]);
fg(x).print();

console.log(tf.memory())