import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"

export const AED = (
    dk: number,
    dv: number,
    bookSize: number,
    down: number = 4
): [
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
    let inp2enc = tf.layers.separableConv2d({
        kernelSize: 3,
        filters: dk,
        padding: "same",
        inputShape: [1, 1, 3],
        trainable: true,
        name: "inp2enc",
    })
    inp2enc.build([null, 1, 1, 3])

    let encoder = [
        tf.layers.separableConv2d({
            kernelSize: 3,
            filters: dk,
            padding: "same",
            inputShape: [1, 1, dk],
            trainable: true,
            name: "enc0",
        }),
        tf.layers.separableConv2d({
            kernelSize: 3,
            filters: dk,
            padding: "same",
            inputShape: [1, 1, dk],
            trainable: true,
            name: "enc1",
        }),
    ]
    encoder.forEach((enc) => enc.build([null, 1, 1, dk]))

    let book = tf.sequential({
        layers: [
            tf.layers.conv2d({
                inputShape: [1, 1, dk],
                filters: bookSize,
                kernelSize: 1,
                useBias: false,
                activation: "softmax",
                trainable: true,
                name: "qk_mapping",
            }),
            tf.layers.conv2d({
                filters: dv,
                kernelSize: 1,
                useBias: false,
                trainable: true,
                name: "value",
            }),
        ],
    })

    let decoder = [
        tf.layers.separableConv2d({
            kernelSize: 3,
            filters: dv,
            padding: "same",
            inputShape: [1, 1, dv],
            trainable: true,
            name: "dec0",
        }),
        tf.layers.separableConv2d({
            kernelSize: 3,
            filters: dv,
            padding: "same",
            inputShape: [1, 1, dv],
            trainable: true,
            name: "dec1",
        }),
    ]
    decoder.forEach((dec) => dec.build([null, 1, 1, dv]))

    let dec2out = tf.layers.separableConv2d({
        kernelSize: 3,
        filters: 3,
        padding: "same",
        inputShape: [1, 1, dv],
        trainable: true,
        name: "dec2out",
    })
    dec2out.build([null, 1, 1, dv])

    let maxPool = <nn.tfFn>((inp: tf.Tensor) => {
        return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, 2, 2, "same")
    })
    let unSampling = tf.layers.upSampling2d({ size: [2, 2], inputShape: [1, 1, dv] })
    return [
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() =>
                    nn.pipe(
                        nn.layerFn(inp2enc),
                        nn.mish,
                        ...new Array(down - 1).fill(
                            nn.pipe(nn.layerFn(encoder[0]), nn.mish, nn.layerFn(encoder[1]), maxPool, nn.mish)
                        ),
                        nn.pipe(nn.layerFn(encoder[0]), nn.mish, nn.layerFn(encoder[1]), maxPool)
                    )(input)
                ),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>inp2enc.getWeights()),
                    ...(<tf.Variable[]>encoder.reduce((prev, enc) => prev.concat(enc.getWeights()), [])),
                ]),
        },
        {
            fn: (input: tf.Tensor) => tf.tidy(() => <tf.Tensor>book.apply(input)),
            ws: () => tf.tidy(() => [...(<tf.Variable[]>book.getWeights())]),
        },
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() =>
                    nn.pipe(
                        ...new Array(down).fill(
                            nn.pipe(
                                nn.layerFn(unSampling),
                                nn.layerFn(decoder[0]),
                                nn.mish,
                                nn.layerFn(decoder[1]),
                                nn.mish
                            )
                        ),
                        nn.layerFn(dec2out)
                    )(input)
                ),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>decoder.reduceRight((prev, decoder) => prev.concat(decoder.getWeights()), [])),
                    ...(<tf.Variable[]>dec2out.getWeights()),
                ]),
        },
    ]
}
