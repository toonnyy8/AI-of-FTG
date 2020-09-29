import * as tf from "@tensorflow/tfjs"

const canvas = <HTMLCanvasElement>document.getElementById("canvas")

const insertPadding = (input: tf.Tensor) => tf.tidy(() => {
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
tf.browser.toPixels(<tf.Tensor3D>insertPadding(tf.fill([50, 50, 1], 1)), canvas)
let l = tf.layers.dense({ inputShape: [5], units: 3 });
(<tf.Tensor>l.apply(tf.fill([50, 50, 50, 5], 1))).print()