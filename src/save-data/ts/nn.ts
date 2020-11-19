import * as tf from "@tensorflow/tfjs"

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

export const stopGradient = tf.customGrad((x, save) => {
    // Save x to make sure it's available later for the gradient.
    ;(<tf.GradSaveFunc>save)([<tf.Tensor>x])
    // Override gradient of our custom x ^ 2 op to be dy * abs(x);
    return {
        value: (<tf.Tensor>x).clone(),
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

export type tfFn = (inp: tf.Tensor) => tf.Tensor

export const pipe = (...fns: ((inp: tf.Tensor) => tf.Tensor)[]): tfFn => {
    return (inp: tf.Tensor) => tf.tidy(() => fns.reduce((prev, fn) => fn(prev), inp))
}

export const layerFn = (layer: tf.layers.Layer): tfFn => (inp: tf.Tensor): tf.Tensor => <tf.Tensor>layer.apply(inp)

export const blurPooling = (filtSize: 1 | 2 | 3 | 4 | 5 | 6 | 7, strides: number) => {
    const kernel: tf.Tensor4D = tf.tidy(() => {
        let k: tf.Tensor = tf.tensor([1, 3, 3, 1], [4, 1])
        switch (filtSize) {
            case 1:
                k = tf.tensor([1], [1, 1])
                break
            case 2:
                k = tf.tensor([1, 1], [2, 1])
                break
            case 3:
                k = tf.tensor([1, 2, 1], [3, 1])
                break
            case 4:
                k = tf.tensor([1, 3, 3, 1], [4, 1])
                break
            case 5:
                k = tf.tensor([1, 4, 6, 4, 1], [5, 1])
                break
            case 6:
                k = tf.tensor([1, 5, 10, 10, 5, 1], [6, 1])
                break
            case 7:
                k = tf.tensor([1, 6, 15, 20, 15, 6, 1], [7, 1])
                break
        }
        k = k.matMul(k, false, true).reshape([filtSize, filtSize, 1, 1])
        return k.div(k.sum())
    })

    return {
        fn: (x: tf.Tensor3D | tf.Tensor4D) =>
            tf.tidy(() => {
                let [batch, h, w, c] = x.shape
                if (c == null) {
                    c = w
                }
                return tf.depthwiseConv2d(x, <tf.Tensor4D>kernel.tile([1, 1, c, 1]), strides, "same")
            }),
        dispose: () => kernel.dispose(),
    }
}
