import * as tf from "@tensorflow/tfjs"

const mish = (x: tf.Tensor) =>
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

export const AED = (layers: number) => {
    const convs = new Array(layers)
        .fill(0)
        .map((_, idx) => tf
            .layers
            .separableConv2d({
                kernelSize: 3,
                filters: 64 * 2 ** Math.floor(idx / 2),
                strides: 2,
                padding: "same"
            })
        )

    const deconvs = new Array(layers)
        .fill(0)
        .map((_, idx) => tf
            .layers
            .separableConv2d({
                kernelSize: 3,
                filters: 64 * 2 ** Math.floor((layers - idx - 1) / 2),
                padding: "same"
            })
        )


    return {
        ae: (input: tf.Tensor3D | tf.Tensor4D) =>
            tf.tidy(() =>
                convs.reduce((inp, conv) =>
                    mish(<tf.Tensor>conv.apply(inp)),
                    input,
                )
            ),
        ae_ws: () =>
            tf.tidy(() =>
                convs.reduce(
                    (w, conv) =>
                        w.concat(...<tf.Variable[]>conv.getWeights()),
                    <tf.Variable[]>[],
                )
            ),

        ad: (input: tf.Tensor3D | tf.Tensor4D) =>
            tf.tidy(() =>
                deconvs.reduce((inp, deconv) =>
                    mish(<tf.Tensor>deconv.apply(insertPadding(inp))),
                    input,
                )
            ),
        ad_ws: () =>
            tf.tidy(() =>
                deconvs.reduce(
                    (w, deconv) =>
                        w.concat(...<tf.Variable[]>deconv.getWeights()),
                    <tf.Variable[]>[],
                )
            ),
    }

}