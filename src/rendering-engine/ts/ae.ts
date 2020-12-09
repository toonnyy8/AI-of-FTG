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

            tf.layers.conv2d({ name: "e2-1", filters: 32, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e2-1_bn" }),
            tf.layers.conv2d({ name: "e2-2", filters: 32, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e2-2_bn" }),
            tf.layers.maxPool2d({ poolSize: 2, strides: 2, padding: "same" }),

            tf.layers.conv2d({ name: "e3-1", filters: 64, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e3-1_bn" }),
            tf.layers.conv2d({ name: "e3-2", filters: 64, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e3-2_bn" }),
            tf.layers.maxPool2d({ poolSize: 2, strides: 2, padding: "same" }),

            // tf.layers.conv2d({ name: "eout", filters: 8, kernelSize: 3, padding: "same" }),
            // layers.mish({}),
            // tf.layers.batchNormalization({ name: "eout_bn" }),

            tf.layers.flatten(),
            tf.layers.dense({ units: 256, name: "ef" }),
            tf.layers.batchNormalization({ name: "ef_bn" }),
            tf.layers.activation({ activation: "sigmoid" }),
            tf.layers.reshape({ targetShape: [8, 32] }),
        ],
    })

    let template = tf.variable(tf.randomNormal([1, 8, 16, 8, 1, 32]), true, "template")
    let dec = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [8, 16, 64], dtype: "float32" }),
            tf.layers.separableConv2d({ name: "din", filters: 64, kernelSize: 3, padding: "same" }),
            layers.mish({}),

            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [32, 64, 16] }),
            layers.mish({}),
            tf.layers.separableConv2d({ name: "d2", filters: 32, kernelSize: 3, padding: "same" }),

            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [32, 64, 8] }),
            layers.mish({}),
            tf.layers.separableConv2d({ name: "d3", filters: 32, kernelSize: 3, padding: "same" }),

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
                    // let a = input
                    let a = tf
                        .mul(template.softmax(-1), input.reshape([-1, 1, 1, 1, 8, 32]))
                        .sum(-1)
                        .reshape([-1, 8, 16, 64])

                    return <tf.Tensor>dec.apply(a)
                }),
            ws: () => tf.tidy(() => [template, ...(<tf.Variable[]>dec.getWeights())]),
        },
    ]
}
