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

export const gaussian = (windowSize: number[], sigma: number) =>
    tf.tidy(() => {
        let dis = windowSize
            .map((size, idx) => {
                if (size % 2 === 0) throw new Error("windowSize must be odd")
                let shape = new Array(windowSize.length).fill(1)
                shape[idx] = size
                return tf
                    .linspace(-(size - 1) / 2, (size - 1) / 2, size)
                    .reshape(shape)
                    .square()
            })
            .reduce((prev, curr) => tf.add(prev, curr))

        let k = dis
            .neg()
            .div(2 * sigma ** 2)
            .exp()
            .div((2 * Math.PI * sigma ** 2) ** 0.5)
        return k.div(k.sum())
    })

export const ssim2d = <T extends tf.Tensor2D | tf.Tensor3D | tf.Tensor4D>(img1: T, img2: T, windowSize: number = 11) =>
    tf.tidy(() => {
        let _img1: tf.Tensor4D | tf.Tensor5D, _img2: tf.Tensor4D | tf.Tensor5D
        const gk = <tf.Tensor5D>gaussian([11, 11, 1, 1, 1], 1.5)
        if (img1.shape.length == 2) {
            _img1 = img1.expandDims(-1).expandDims(-1)
            _img2 = img2.expandDims(-1).expandDims(-1)
        } else if (img1.shape.length == 3) {
            _img1 = img1.expandDims(-1)
            _img2 = img2.expandDims(-1)
        } else if (img1.shape.length == 4) {
            _img1 = img1.expandDims(-1)
            _img2 = img2.expandDims(-1)
        } else {
            throw new Error("Invalid image dimension")
        }
        const mu1 = tf.conv3d(_img1, gk, 1, "same")
        const mu2 = tf.conv3d(_img2, gk, 1, "same")

        const mu1_sq = mu1.square()
        const mu2_sq = mu2.square()
        const mu1_mu2 = mu1.mul(mu2)

        const sigma1_sq = tf.conv3d(_img1.square(), gk, 1, "same").sub(mu1_sq)
        const sigma2_sq = tf.conv3d(_img2.square(), gk, 1, "same").sub(mu2_sq)
        const sigma12 = tf.conv3d(<tf.Tensor4D | tf.Tensor5D>_img1.mul(_img2), gk, 1, "same").sub(mu1_mu2)

        const C1 = 0.01 ** 2
        const C2 = 0.03 ** 2

        const ssim_map = mu1_mu2
            .mul(2)
            .add(C1)
            .mul(sigma12.mul(2).add(C2))
            .div(mu1_sq.add(mu2_sq).add(C1).mul(sigma1_sq.add(sigma2_sq).add(C2)))

        if (img1.shape.length == 2 || img1.shape.length == 3) {
            return <tf.Scalar>ssim_map.mean()
        } else {
            return <tf.Tensor1D>ssim_map.mean([1, 2, 3, 4])
        }
    })
