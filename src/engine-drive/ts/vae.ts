import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"

const c2pix = (t: tf.Tensor4D) => {
    let [batch, h, w, c] = t.shape
    return <tf.Tensor4D>t
        .transpose([0, 2, 1, 3])
        .reshape([batch, w, h * 2, c / 2])
        .transpose([0, 2, 1, 3])
        .reshape([batch, h * 2, w * 2, c / 4])
}

export const VAE = (
    config: {} = {}
): {
    enc: {
        fn: (input: tf.Tensor4D) => { mu: tf.Tensor2D; logvar: tf.Tensor2D }
        ws: () => tf.Variable[]
    }
    reparametrize: <T extends tf.Tensor>(mu: T, logvar: T) => T
    dec: {
        fn: (input: tf.Tensor2D, random?: boolean) => tf.Tensor4D
        synthesizer: (input: tf.Tensor2D) => tf.Tensor4D
        ws: () => tf.Variable[]
    }
} => {
    let blurPooling = nn.blurPooling(3, 2)
    let pooling = (a: tf.Tensor) => {
        return blurPooling.fn(tf.maxPool(<tf.Tensor4D | tf.Tensor3D>a, 2, 1, "same"))
    }
    let enc1 = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [32, 64, 3] }),
            tf.layers.conv2d({ name: "ein", filters: 32, kernelSize: 1, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "ein_bn" }),

            tf.layers.maxPooling2d({ poolSize: 2, strides: 1, padding: "same" }),

            tf.layers.conv2d({ name: "e1-1", filters: 32, kernelSize: 3, strides: 2, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e1-1_bn" }),

            tf.layers.maxPooling2d({ poolSize: 2, strides: 1, padding: "same" }),

            tf.layers.conv2d({ name: "e2-1", filters: 64, kernelSize: 3, strides: 2, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e2-1_bn" }),
        ],
    })

    let enc2 = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [8, 16, 64] }),

            tf.layers.maxPooling2d({ poolSize: 2, strides: 1, padding: "same" }),

            tf.layers.conv2d({ name: "e3-1", filters: 128, kernelSize: 3, strides: 2, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e3-1_bn" }),

            tf.layers.maxPooling2d({ poolSize: 2, strides: 1, padding: "same" }),

            tf.layers.conv2d({ name: "e4-1", filters: 256, kernelSize: 3, strides: 2, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "e4-1_bn" }),
        ],
    })

    let enc_spatial = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [8, 16, 64] }),

            tf.layers.conv2d({ name: "enc_spatial", filters: 1, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "enc_spatial_bn" }),
        ],
    })
    let enc_channel = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [2, 4, 256] }),

            tf.layers.conv2d({ name: "enc_channel", filters: 32, kernelSize: 3, padding: "same" }),
            layers.mish({}),
            tf.layers.batchNormalization({ name: "enc_channel_bn" }),
        ],
    })
    let muAndLogvar = tf.sequential({
        layers: [tf.layers.inputLayer({ inputShape: [384] }), tf.layers.dense({ name: "muAndLogvar", units: 256 * 2 })],
    })
    let unzip = tf.sequential({
        layers: [tf.layers.inputLayer({ inputShape: [256] }), tf.layers.dense({ name: "unzip", units: 384 })],
    })
    let synthesizerIn = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [2, 4, 32], dtype: "float32" }),
            tf.layers.conv2d({ name: "synthesizer", filters: 256, kernelSize: 1, padding: "same" }),
        ],
    })

    let dec = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [8, 16, 64], dtype: "float32" }),
            tf.layers.separableConv2d({ name: "d1", filters: 128, kernelSize: 3, padding: "same" }),

            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [32, 64, 32] }),
            layers.mish({}),
            tf.layers.separableConv2d({ name: "d2", filters: 128, kernelSize: 3, padding: "same" }),

            layers.lambda({ fn: (t) => c2pix(<tf.Tensor4D>t), outputShape: [32, 64, 32] }),
            layers.mish({}),
            tf.layers.separableConv2d({ name: "d3", filters: 64, kernelSize: 3, padding: "same" }),

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
    const synthesizer = (input: tf.Tensor2D): tf.Tensor4D => {
        let [batch] = input.shape
        let [_spatial, _channel] = <tf.Tensor[]>input.split([128, 256], -1)
        let spatial = <tf.Tensor4D>_spatial.reshape([batch, 8, 16, 1])
        let channel = <tf.Tensor4D>_channel.reshape([batch, 2, 4, 32])
        channel = <tf.Tensor4D>synthesizerIn.apply(channel)
        channel = tf.image.resizeBilinear(c2pix(channel), [8, 16])
        return tf.mul(spatial, channel)
    }

    return {
        enc: {
            fn: (input: tf.Tensor4D) =>
                tf.tidy(() => {
                    let [batch] = input.shape
                    let a = <tf.Tensor>enc1.apply(input)
                    let b = <tf.Tensor>enc2.apply(a)
                    let spatial = <tf.Tensor>enc_spatial.apply(a)
                    let channel = <tf.Tensor>enc_channel.apply(b)
                    let out = <tf.Tensor2D>tf.concat([spatial.reshape([batch, -1]), channel.reshape([batch, -1])], -1)
                    out = <tf.Tensor2D>muAndLogvar.apply(out)
                    let [mu, logvar] = <[tf.Tensor2D, tf.Tensor2D]>out.split(2, -1)
                    return { mu, logvar }
                }),
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>enc1.getWeights()),
                    ...(<tf.Variable[]>enc2.getWeights()),
                    ...(<tf.Variable[]>enc_spatial.getWeights()),
                    ...(<tf.Variable[]>enc_channel.getWeights()),
                    ...(<tf.Variable[]>muAndLogvar.getWeights()),
                ]),
        },
        reparametrize: <T extends tf.Tensor>(mu: T, logvar: T) => {
            let std = logvar.mul(0.5).exp()
            let eps = tf.truncatedNormal(mu.shape, 0, 1)
            return eps.mul(std).add(mu)
        },
        dec: {
            fn: (input: tf.Tensor2D) =>
                tf.tidy(() => {
                    let inp = synthesizer(<tf.Tensor2D>unzip.apply(input))
                    // let inp = synthesizer(input)
                    let out = <tf.Tensor4D>dec.apply(inp)
                    return out
                }),
            synthesizer,
            ws: () =>
                tf.tidy(() => [
                    ...(<tf.Variable[]>unzip.getWeights()),
                    ...(<tf.Variable[]>synthesizerIn.getWeights()),
                    ...(<tf.Variable[]>dec.getWeights()),
                ]),
        },
    }
}
