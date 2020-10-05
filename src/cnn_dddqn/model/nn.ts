import * as tf from "@tensorflow/tfjs"
import { positionalEncoding } from "./mha"

export const mish = (x: tf.Tensor) => tf.tidy(() => tf.mul(x, tf.tanh(tf.softplus(x))))

export const insertPadding = (input: tf.Tensor) =>
    tf.tidy(() => {
        switch (input.shape.length) {
            case 3: {
                const [h, w, c] = input.shape
                const horizontal = input.concat([tf.fill([h, w, c], 0)], 1).reshape([h * 2, w, c])
                const vertical = horizontal.concat([tf.fill([h * 2, w, c], 0)], 2).reshape([h * 2, w * 2, c])
                return vertical
            }
            case 4: {
                const [b, h, w, c] = input.shape
                const horizontal = input.concat([tf.fill([b, h, w, c], 0)], 2).reshape([b, h * 2, w, c])
                const vertical = horizontal.concat([tf.fill([b, h * 2, w, c], 0)], 3).reshape([b, h * 2, w * 2, c])
                return vertical
            }
        }
    })

export const unPooling = (input: tf.Tensor) =>
    tf.tidy(() => {
        switch (input.shape.length) {
            case 3: {
                const [h, w, c] = input.shape
                const horizontal = input.concat([input], 1).reshape([h * 2, w, c])
                const vertical = horizontal.concat([horizontal], 2).reshape([h * 2, w * 2, c])
                return vertical
            }
            case 4: {
                const [b, h, w, c] = input.shape
                const horizontal = input.concat([input], 2).reshape([b, h * 2, w, c])
                const vertical = horizontal.concat([horizontal], 3).reshape([b, h * 2, w * 2, c])
                return vertical
            }
        }
    })

export const stopGradient = tf.customGrad((x: tf.Tensor, save: tf.GradSaveFunc) => {
    // Save x to make sure it's available later for the gradient.
    save([x])
    // Override gradient of our custom x ^ 2 op to be dy * abs(x);
    return {
        value: x.clone(),
        // Note `saved.x` which points to the `x` we saved earlier.
        gradFunc: (dy, saved) => {
            return [tf.mul(dy, 0)]
        },
    }
})

// export const positionPooling = (
//     x: tf.Tensor4D,
//     filterSize: number | [number, number],
//     strides: number | [number, number],
//     pad: number | "valid" | "same"
// ) =>
//     tf.tidy(() => {
//         let [batch, h, w, c] = x.shape
//         let peH = positionalEncoding(h, c).expandDims(2).tile([1, 1, w, 1])
//         let peW = positionalEncoding(w, c).expandDims(1).tile([1, h, 1, 1])
//         let pe = tf.add(peH, peW)
//         const { result, indexes } = tf.maxPoolWithArgmax(x, filterSize, strides, pad)
//         return tf.add(result, pe.flatten().gather(indexes.flatten().cast("int32")).reshapeAs(result))
//     })
