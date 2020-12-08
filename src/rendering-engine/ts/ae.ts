import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"

const pix2c = (t: tf.Tensor4D) => {
    let [batch, h, w, c] = t.shape
    return <tf.Tensor4D>t
        .reshape([batch, h, w / 2, c * 2])
        .transpose([0, 2, 1, 3])
        .reshape([batch, w / 2, h / 2, c * 4])
        .transpose([0, 2, 1, 3])
}
const c2pix = (t: tf.Tensor4D) => {
    let [batch, h, w, c] = t.shape
    return <tf.Tensor4D>t
        .transpose([0, 2, 1, 3])
        .reshape([batch, w, h * 2, c / 2])
        .transpose([0, 2, 1, 3])
        .reshape([batch, h * 2, w * 2, c / 4])
}
const c2pix_1_2 = (t: tf.Tensor4D) => {
    let [batch, h, w, c] = t.shape
    return <tf.Tensor4D>t.reshape([batch, h, w * 2, c / 2])
}
const c2pix_2_1 = (t: tf.Tensor4D) => {
    let [batch, h, w, c] = t.shape
    return <tf.Tensor4D>t
        .transpose([0, 2, 1, 3])
        .reshape([batch, w, h * 2, c / 2])
        .transpose([0, 2, 1, 3])
}
const mirrorPad = (outputShape: tf.Shape) => {
    return layers.lambda({
        fn: (t) =>
            tf.mirrorPad(
                t,
                [
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [0, 0],
                ],
                "reflect"
            ),
        outputShape,
    })
}

export const AED = (
    config: {
        assetGroups: number
        assetSize: number
        assetNum: number
        dk: number
    } = { assetGroups: 4, assetSize: 16, assetNum: 32, dk: 8 }
): [
    {
        fn: (input: tf.Tensor) => tf.Tensor
        ws: () => tf.Variable[]
    },
    {
        fn: (input: tf.Tensor) => tf.Tensor
        ws: () => tf.Variable[]
    }
] => {
    let blurPooling = nn.blurPooling(3, 2)
    let encRes = (a: tf.Tensor, b: tf.Tensor) => {
        return blurPooling.fn(tf.add(a, tf.maxPool(<tf.Tensor4D | tf.Tensor3D>b, 2, 1, "same")))
    }
    let pooling = (a: tf.Tensor) => {
        return blurPooling.fn(tf.maxPool(<tf.Tensor4D | tf.Tensor3D>a, 2, 1, "same"))
    }
    let enc = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [32, 64, 3] }),
            tf.layers.conv2d({ name: "ein", filters: 32, kernelSize: 1, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "ein_bn" }),

            tf.layers.conv2d({ name: "e1-1", filters: 32, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e1-1_bn" }),
            tf.layers.conv2d({ name: "e1-2", filters: 32, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e1-2_bn" }),
            tf.layers.maxPool2d({ poolSize: 2, strides: 2, padding: "same" }),

            tf.layers.reshape({ targetShape: [16, 32 * 32] }),
            tf.layers.dense({ name: "ew", units: 8 * 32 }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "ew_bn" }),
            tf.layers.reshape({ targetShape: [16, 8, 32] }),

            tf.layers.permute({ dims: [2, 1, 3] }),

            tf.layers.reshape({ targetShape: [8, 16 * 32] }),
            tf.layers.dense({ name: "eh", units: 4 * 32 }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "eh_bn" }),
            tf.layers.reshape({ targetShape: [8, 4, 32] }),

            tf.layers.permute({ dims: [2, 1, 3] }),

            // tf.layers.dense({ name: "ed", units: 8 }),
            // layers.mish({}),
            // tf.layers.batchNormalization({ name: "ed_bn" }),
        ],
    })

    let dec = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [16, 8, 32], dtype: "float32" }),

            // tf.layers.dense({ name: "dd", units: 32 }),

            tf.layers.permute({ dims: [2, 1, 3] }),

            tf.layers.reshape({ targetShape: [8, 4 * 32] }),
            layers.mish({}),
            tf.layers.dense({ name: "dh", units: 16 * 32 }),
            tf.layers.reshape({ targetShape: [8, 16, 32] }),

            tf.layers.permute({ dims: [2, 1, 3] }),

            tf.layers.reshape({ targetShape: [16, 8 * 32] }),
            layers.mish({}),
            tf.layers.dense({ name: "dw", units: 32 * 32 }),
            tf.layers.reshape({ targetShape: [16, 32, 32] }),

            layers.mish({}),
            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [32, 64, 8] }),
            tf.layers.separableConv2d({ name: "d3", filters: 32, kernelSize: 3, padding: "same" }),

            // layers.mish({}),
            // layers.lambda({ fn: (t) => c2pix_2_1(<tf.Tensor4D>t), outputShape: [64, 64, 16] }),
            // tf.layers.separableConv2d({ name: "d4", filters: 32, kernelSize: 3, padding: "same" }),

            layers.mish({}),
            tf.layers.conv2d({
                name: "dout",
                filters: 3,
                kernelSize: 3,
                padding: "same",
                activation: "sigmoid",
            }),
        ],
    })

    return [
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() => {
                    let out = <tf.Tensor>enc.apply(input)
                    // let v = <tf.Tensor>enc_v.apply(out)
                    // let h = <tf.Tensor>enc_h.apply(out)
                    // let vh = tf.concat([v, h], -1)
                    return out.sigmoid() //(<tf.Tensor>enc_vh.apply(vh)).add(vh).sigmoid() //out_vh
                }),
            ws: () => tf.tidy(() => [...(<tf.Variable[]>enc.getWeights())]),
        },
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() => {
                    // let [v, h] = input.split(2, -1)
                    // v = <tf.Tensor>dec_v.apply(v.reshape([-1, 1, 16, 16]))
                    // h = <tf.Tensor>dec_h.apply(h.reshape([-1, 8, 1, 32]))
                    // let temp = tf.concat([v, h], -1)
                    return <tf.Tensor>dec.apply(input)
                }),
            ws: () => tf.tidy(() => [...(<tf.Variable[]>dec.getWeights())]),
        },
    ]
}
