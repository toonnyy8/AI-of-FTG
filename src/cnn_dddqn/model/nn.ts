import * as tf from "@tensorflow/tfjs"

export const mish = (x: tf.Tensor) =>
    tf.tidy(() => tf.mul(x, tf.tanh(tf.softplus(x))))

export const insertPadding = (input: tf.Tensor) => tf.tidy(() => {
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

export const unPooling = (input: tf.Tensor) => tf.tidy(() => {
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