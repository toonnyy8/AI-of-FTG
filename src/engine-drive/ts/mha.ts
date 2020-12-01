import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"

const getAngles = (pos: tf.Tensor, i: tf.Tensor, d_model: number) =>
    tf.tidy(() => {
        const angleRates = tf.div(1, tf.pow(10000, tf.mul(tf.floorDiv(i, 2), 2 / d_model)))
        return pos.matMul(angleRates)
    })

export const positionalEncoding = (position: number, d_model: number) => {
    if (d_model % 2 == 0) {
        const angleRads = getAngles(
            tf.linspace(0, position - 1, position).expandDims(1),
            tf.linspace(0, d_model - 1, d_model).expandDims(0),
            d_model
        )
        const [even, odd] = angleRads.reshape([position, -1, 2]).unstack(-1)

        return tf.stack([tf.sin(even), tf.cos(odd)], -1).reshape([1, position, -1])
    } else {
        const angleRads = getAngles(
            tf.linspace(0, position - 1, position).expandDims(1),
            tf.linspace(0, d_model, d_model + 1).expandDims(0),
            d_model
        )
        const [even, odd] = angleRads.reshape([position, -1, 2]).unstack(-1)

        return tf
            .stack([tf.sin(even), tf.cos(odd)], -1)
            .reshape([1, position, -1])
            .slice([0, 0, 0], [-1, -1, d_model])
    }
}

export const MHA = (d_model: number, h: number, dk: number, dv: number) => {
    const QLinear = tf.layers.dense({ inputShape: [d_model], units: dk * h })
    const KVLinear = tf.layers.dense({
        inputShape: [d_model],
        units: (dk + dv) * h,
    })

    const outLinear = tf.layers.dense({ inputShape: [dv * h], units: d_model })

    return {
        fn: (Qin: tf.Tensor2D, KVin: tf.Tensor2D) =>
            tf.tidy(() => {
                const [Lq] = Qin.shape
                const [Lkv] = KVin.shape
                const mask = tf
                    .sub(1, tf.linalg.bandPart(tf.ones([Lq, Lkv]), -1, 0))
                    .mul(-1e9)
                    .expandDims(-1)

                let reshape2Head = (t: tf.Tensor2D) => {
                    const [l] = t.shape
                    return <tf.Tensor3D>t.reshape([l, h, -1]).transpose([1, 0, 2])
                }
                const Q = <tf.Tensor2D>QLinear.apply(Qin)

                const [K, V] = <tf.Tensor2D[]>(<tf.Tensor>KVLinear.apply(KVin)).split([dk * h, dv * h], -1)

                const att = nn.softmax(
                    tf
                        .mul(Q.reshape([Lq, 1, h, dk]), K.reshape([1, Lkv, h, dk]))
                        .sum(-1)
                        .div(dk ** 0.5)
                        .add(mask),
                    1
                )

                let headConcat = (t: tf.Tensor) => {
                    const [, l] = t.shape
                    return t.transpose([1, 0, 2]).reshape([l, -1])
                }

                return <tf.Tensor2D>outLinear.apply(
                    tf
                        .mul(att.reshape([Lq, Lkv, h, 1]), V.reshape([1, Lkv, h, dv]))
                        .sum(1)
                        .reshape([Lq, h * dv])
                )
            }),
        ws: () => <tf.Variable[]>[...QLinear.getWeights(), ...KVLinear.getWeights(), ...outLinear.getWeights()],
    }
}

export const FF = (d_model: number, hiddens: number) => {
    const linear1 = tf.layers.dense({ inputShape: [d_model], units: hiddens })
    const linear2 = tf.layers.dense({ inputShape: [hiddens], units: d_model })
    return {
        fn: (x: tf.Tensor) => linear2.apply(nn.mish(<tf.Tensor>linear1.apply(x))),
        ws: () => <tf.Variable[]>[...linear1.getWeights(), ...linear2.getWeights()],
    }
}
