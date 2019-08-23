import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim

import * as transformerXL from "./MirageNet/transformerXL"
console.log(transformerXL)
console.log(tf.memory())

const customOp = tf.customGrad((x, y, save) => {
    return tf.tidy(() => {
        // Save x to make sure it's available later for the gradient.
        save([x, y]);
        // Override gradient of our custom x ^ 2 op to be dy * abs(x);
        return {
            value: tf.mul(x, y).square(),
            // Note `saved.x` which points to the `x` we saved earlier.
            gradFunc: (dy, saved) => [tf.zeros(saved[0].shape), tf.zeros(saved[1].shape)]
        };
    })
});

const x = tf.tensor1d([-1, -2, 3]);
const y = tf.tensor1d([-1, 0.8, 1.5]);

let f = (x, y) => { return customOp(x, y) }
const dx = tf.grads((x, y) => customOp(x, y));

console.log(`f(x,y):`);
f(x, y).print();
console.log(`f'(x,y):`);
// dx([x, y])[0].print();

console.log(tf.memory())