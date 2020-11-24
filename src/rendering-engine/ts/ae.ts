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
    } = { assetGroups: 4, assetSize: 16, assetNum: 32 }
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
    let inLayer = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [40, 40, 3] }),
            tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization(),
        ],
    })
    let encs = [
        tf.sequential({
            layers: [
                tf.layers.inputLayer({ inputShape: [40, 40, 64] }),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.inputLayer({ inputShape: [20, 20, 64] }),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.inputLayer({ inputShape: [10, 10, 64] }),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.inputLayer({ inputShape: [5, 5, 64] }),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 64, padding: "same" }),
                layers.mish({}),
                tf.layers.batchNormalization(),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.inputLayer({ inputShape: [5, 5, 64] }),
                tf.layers.separableConv2d({ kernelSize: 3, filters: 4, padding: "same" }),
                layers.mish({}),
            ],
        }),
    ]

    let mappingKeys = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [5, 5, 4], dtype: "float32" }),
            tf.layers.separableConv2d({
                name: "mappingKeys",
                kernelSize: 3,
                filters: config.assetNum * config.assetGroups,
                padding: "same",
                useBias: false,
            }),
        ],
    })
    let assets = tf.variable(
        tf.randomNormal([1, 1, 1, config.assetGroups, config.assetNum, config.assetSize]),
        true,
        "assets"
    )

    let dec = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [5, 5, config.assetGroups * config.assetSize], dtype: "float32" }),
            layers.lambda({
                fn: (t) => c2pix(<tf.Tensor4D>t),
                outputShape: [10, 10, (config.assetGroups * config.assetSize) / 4],
            }),

            tf.layers.separableConv2d({
                name: "decoderConv1",
                kernelSize: 3,
                filters: 128,
                padding: "same",
            }),
            layers.mish({}),
            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [20, 20, 32] }),

            tf.layers.separableConv2d({
                name: "decoderConv3",
                kernelSize: 3,
                filters: 128,
                padding: "same",
            }),
            layers.mish({}),
            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [40, 40, 32] }),

            tf.layers.separableConv2d({
                name: "decoderConv4",
                kernelSize: 3,
                filters: 3,
                padding: "same",
                activation: "sigmoid",
            }),
        ],
    })
    return [
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() => {
                    let a = <tf.Tensor>inLayer.apply(input)
                    let b = <tf.Tensor>encs[0].apply(tf.add(a, <tf.Tensor>encs[0].apply(a)))
                    a = encRes(a, b)
                    b = <tf.Tensor>encs[1].apply(tf.add(a, <tf.Tensor>encs[1].apply(a)))
                    a = encRes(a, b)
                    b = <tf.Tensor>encs[2].apply(tf.add(a, <tf.Tensor>encs[2].apply(a)))
                    a = encRes(a, b)
                    b = <tf.Tensor>encs[3].apply(tf.add(a, <tf.Tensor>encs[3].apply(a)))
                    a = tf.add(a, b)
                    let out = <tf.Tensor>encs[4].apply(a)
                    return out
                }),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>inLayer.getWeights()),
                    ...encs.reduce((prev, enc) => prev.concat(<tf.Variable[]>enc.getWeights()), <tf.Variable[]>[]),
                ]),
        },
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() => {
                    const [b, h, w, c] = input.shape
                    const att = (<tf.Tensor4D>mappingKeys.apply(input))
                        .reshape([b, h, w, config.assetGroups, config.assetNum])
                        .softmax(4)
                        .expandDims(-1)
                    return <tf.Tensor>dec.apply(
                        tf
                            .mul(assets, att)
                            .sum(4)
                            .reshape([b, h, w, -1])
                    )
                }),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>mappingKeys.getWeights()),
                    assets,
                    ...(<tf.Variable[]>dec.getWeights()),
                ]),
        },
    ]
}
