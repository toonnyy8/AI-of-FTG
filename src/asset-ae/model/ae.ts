import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"

export const AED = ({
    dh = 4,
    dk = 8,
    assetSize = 64,
    assetNum = 256,
    assetGroups = 4,
    down = 4,
    pureAED = false,
}: {
    dh?: number
    dk?: number
    assetSize?: number
    assetNum?: number
    assetGroups?: number
    down?: number
    pureAED?: boolean
}): [
    {
        fn: (input: tf.Tensor) => tf.Tensor
        ws: () => tf.Variable[]
    },
    {
        fn: (input: tf.Tensor) => tf.Tensor
        ws: () => tf.Variable[]
    },
    {
        fn: (input: tf.Tensor) => tf.Tensor
        ws: () => tf.Variable[]
    }
] => {
    let inp2enc = tf.sequential({
        layers: [
            tf.layers.separableConv2d({
                kernelSize: 3,
                filters: dh,
                padding: "same",
                inputShape: [1, 1, 3],
                trainable: true,
                name: "inp2enc",
            }),
        ],
    })

    let encoder = new Array(down).fill(0).map((_, idx) => [
        tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dh,
                    padding: "same",
                    inputShape: [1, 1, dh],
                    trainable: true,
                    name: `enc${idx}-0`,
                }),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dh,
                    padding: "same",
                    inputShape: [1, 1, dh],
                    trainable: true,
                    name: `enc${idx}-1`,
                }),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dh,
                    padding: "same",
                    inputShape: [1, 1, dh],
                    trainable: true,
                    name: `enc${idx}-2`,
                }),
            ],
        }),
    ])

    let hv2Qs = new Array(assetGroups).fill(0).map((_, idx) =>
        tf.sequential({
            layers: [
                tf.layers.conv2d({
                    kernelSize: 3,
                    filters: dk,
                    padding: "same",
                    inputShape: [1, 1, dh],
                    trainable: true,
                    name: `hv2q${idx}`,
                }),
            ],
        })
    )
    let assets = new Array(assetGroups)
        .fill(0)
        .map((_, idx) => tf.variable(tf.randomNormal([assetNum, assetSize]), true, `asset${idx}`))
    let asset2Ks = new Array(assetGroups).fill(0).map((_, idx) =>
        tf.sequential({
            layers: [
                tf.layers.dense({
                    units: dk,
                    inputShape: [1, 1, assetSize],
                    trainable: true,
                    name: `asset2k${idx}`,
                }),
            ],
        })
    )
    // pureAED only
    let enc2dec = tf.sequential({
        layers: [
            tf.layers.separableConv2d({
                kernelSize: 3,
                filters: assetGroups * assetSize,
                padding: "same",
                inputShape: [1, 1, dh],
                trainable: true,
                name: "enc2dec",
            }),
        ],
    })
    let decoder = new Array(down).fill(0).map((_, idx) => [
        tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: assetGroups * assetSize,
                    padding: "same",
                    inputShape: [1, 1, assetGroups * assetSize],
                    trainable: true,
                    name: `dec${idx}-0`,
                }),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: assetGroups * assetSize,
                    padding: "same",
                    inputShape: [1, 1, assetGroups * assetSize],
                    trainable: true,
                    name: `dec${idx}-1`,
                }),
            ],
        }),
        tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: assetGroups * assetSize,
                    padding: "same",
                    inputShape: [1, 1, assetGroups * assetSize],
                    trainable: true,
                    name: `dec${idx}-2`,
                }),
            ],
        }),
    ])
    // let decoder = [
    //     tf.sequential({
    //         layers: [
    //             tf.layers.separableConv2d({
    //                 kernelSize: 3,
    //                 filters: assetGroups * assetSize,
    //                 padding: "same",
    //                 inputShape: [1, 1, assetGroups * assetSize],
    //                 trainable: true,
    //                 name: "dec0",
    //             }),
    //         ],
    //     }),
    //     tf.sequential({
    //         layers: [
    //             tf.layers.separableConv2d({
    //                 kernelSize: 3,
    //                 filters: assetGroups * assetSize,
    //                 padding: "same",
    //                 inputShape: [1, 1, assetGroups * assetSize],
    //                 trainable: true,
    //                 name: "dec1",
    //             }),
    //         ],
    //     }),
    //     tf.sequential({
    //         layers: [
    //             tf.layers.separableConv2d({
    //                 kernelSize: 3,
    //                 filters: assetGroups * assetSize,
    //                 padding: "same",
    //                 inputShape: [1, 1, assetGroups * assetSize],
    //                 trainable: true,
    //                 name: "dec2",
    //             }),
    //         ],
    //     }),
    // ]

    let dec2out = tf.sequential({
        layers: [
            tf.layers.separableConv2d({
                kernelSize: 3,
                filters: 3,
                padding: "same",
                inputShape: [1, 1, assetGroups * assetSize],
                trainable: true,
                name: "dec2out",
            }),
        ],
    })

    // let maxPool = <nn.tfFn>((inp: tf.Tensor) => {
    //     return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, 2, 2, "same")
    // })
    let maxPool = <nn.tfFn>((inp: tf.Tensor) => {
        return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, 2, 1, "same")
    })
    let blurPooling = nn.blurPooling(3, 2)
    let unSampling = tf.layers.upSampling2d({ size: [2, 2], inputShape: [1, 1, assetGroups * assetSize] })
    return [
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() =>
                    nn.pipe(
                        nn.layerFn(inp2enc),
                        nn.mish,
                        ...encoder.map((enc) =>
                            nn.pipe(
                                nn.layerFn(enc[0]),
                                nn.mish,
                                nn.layerFn(enc[1]),
                                nn.mish,
                                nn.layerFn(enc[2]),
                                nn.mish,
                                maxPool,
                                blurPooling.fn
                            )
                        )
                    )(input)
                ),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>inp2enc.getWeights()),
                    ...(<tf.Variable[]>(
                        encoder.reduce(
                            (prev, enc) => prev.concat(enc.reduce((prev, _enc) => prev.concat(_enc.getWeights()), [])),
                            <tf.Variable[]>[]
                        )
                    )),
                ]),
        },
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() => {
                    if (pureAED) return nn.mish(nn.layerFn(enc2dec)(input))
                    else if (assetGroups > 1)
                        return tf.concat(
                            new Array(assetGroups).fill(0).map((_, idx) =>
                                tf
                                    .mul(
                                        nn.layerFn(hv2Qs[idx])(input).expandDims(3),
                                        nn.layerFn(asset2Ks[idx])(assets[idx]).reshape([1, 1, 1, assetNum, dk])
                                    )
                                    .sum(-1)
                                    .softmax(-1)
                                    .expandDims(-1)
                                    .mul(assets[idx].reshape([1, 1, 1, assetNum, assetSize]))
                                    .sum(3)
                            ),
                            -1
                        )
                    else
                        return tf
                            .mul(
                                nn.layerFn(hv2Qs[0])(input).expandDims(3),
                                nn.layerFn(asset2Ks[0])(assets[0]).reshape([1, 1, 1, assetNum, dk])
                            )
                            .sum(-1)
                            .softmax(-1)
                            .expandDims(-1)
                            .mul(assets[0].reshape([1, 1, 1, assetNum, assetSize]))
                            .sum(3)
                }),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>hv2Qs.reduce((prev, hv2Q) => prev.concat(hv2Q.getWeights()), [])),
                    ...(<tf.Variable[]>assets),
                    ...(<tf.Variable[]>asset2Ks.reduce((prev, asset2K) => prev.concat(asset2K.getWeights()), [])),
                    ...(<tf.Variable[]>enc2dec.getWeights()),
                ]),
        },
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() =>
                    nn.pipe(
                        ...decoder.map((dec) =>
                            nn.pipe(
                                nn.layerFn(unSampling),
                                nn.layerFn(dec[0]),
                                nn.mish,
                                nn.layerFn(dec[1]),
                                nn.mish,
                                nn.layerFn(dec[2]),
                                nn.mish
                            )
                        ),
                        nn.layerFn(dec2out),
                        tf.sigmoid
                    )(input)
                ),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>(
                        decoder.reduce(
                            (prev, dec) => prev.concat(dec.reduce((prev, _dec) => prev.concat(_dec.getWeights()), [])),
                            <tf.Variable[]>[]
                        )
                    )),
                    ...(<tf.Variable[]>dec2out.getWeights()),
                ]),
        },
    ]
}
