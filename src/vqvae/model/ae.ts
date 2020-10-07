import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"

export const AED = (
    channels: number[],
    zipSize: boolean = true
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
    const enChannels = [...channels]
    const deChannels = [...channels].reverse()

    console.log(enChannels, deChannels)

    const convs = enChannels.slice(1).map((f, idx) =>
        tf.layers.separableConv2d({
            kernelSize: 3,
            filters: f,
            inputShape: [1, 1, 1, enChannels[idx]],
            strides: 1,
            padding: "same",
            trainable: true,
        })
    )
    convs.forEach((conv, idx) => conv.build([1, 1, 1, enChannels[idx]]))

    const deconvs = deChannels.slice(1).map((f, idx) =>
        tf.layers.separableConv2d({
            kernelSize: 3,
            filters: f,
            padding: "same",
            trainable: true,
        })
    )
    deconvs.forEach((deconv, idx) => deconv.build([1, 1, 1, deChannels[idx]]))

    return [
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() =>
                    convs.reduce(
                        (inp, conv) =>
                            zipSize
                                ? tf.maxPool(<tf.Tensor4D>nn.mish(<tf.Tensor>conv.apply(inp)), [2, 2], 2, "same")
                                : nn.mish(<tf.Tensor>conv.apply(inp)),
                        input
                    )
                ),
            ws: () =>
                tf.tidy(() =>
                    convs.reduce((w, conv) => w.concat(...(<tf.Variable[]>conv.getWeights())), <tf.Variable[]>[])
                ),
        },
        {
            fn: (input: tf.Tensor) =>
                tf.tidy(() =>
                    deconvs.reduce((inp, deconv, l) => {
                        if (l < channels.length - 2)
                            return nn.mish(<tf.Tensor>deconv.apply(zipSize ? nn.unPooling(inp) : inp))
                        else return <tf.Tensor>deconv.apply(zipSize ? nn.unPooling(inp) : inp)
                    }, input)
                ),
            ws: () =>
                tf.tidy(() =>
                    deconvs.reduce((w, deconv) => w.concat(...(<tf.Variable[]>deconv.getWeights())), <tf.Variable[]>[])
                ),
        },
    ]
}
