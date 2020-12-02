import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"

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

        return <tf.Tensor2D>tf.stack([tf.sin(even), tf.cos(odd)], -1).reshape([position, -1])
    } else {
        const angleRads = getAngles(
            tf.linspace(0, position - 1, position).expandDims(1),
            tf.linspace(0, d_model, d_model + 1).expandDims(0),
            d_model
        )
        const [even, odd] = angleRads.reshape([position, -1, 2]).unstack(-1)

        return <tf.Tensor2D>tf
            .stack([tf.sin(even), tf.cos(odd)], -1)
            .reshape([position, -1])
            .slice([0, 0], [-1, d_model])
    }
}

export const MHA = (d_model: number, h: number, dk: number, dv: number) => {
    const QLinear = tf.sequential({
        layers: [tf.layers.inputLayer({ inputShape: [d_model] }), tf.layers.dense({ units: dk * h, name: "QLinear" })],
    })
    const KVLinear = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [d_model] }),
            tf.layers.dense({ units: (dk + dv) * h, name: "KVLinear" }),
        ],
    })

    const outLinear = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [dv * h] }),
            tf.layers.dense({ units: d_model, name: "outLinear" }),
        ],
    })

    return {
        fn: (Qin: tf.Tensor2D, KVin: tf.Tensor2D) =>
            tf.tidy(() => {
                const [Lq] = Qin.shape
                const [Lkv] = KVin.shape
                const mask = tf
                    .sub(1, tf.linalg.bandPart(tf.ones([Lq, Lkv]), -1, 0))
                    .mul(-1e9)
                    .expandDims(-1)

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
    const ff = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [d_model] }),
            tf.layers.dense({ units: hiddens, name: "FF-hidden" }),
            layers.mish({}),
            tf.layers.dense({ units: d_model, name: "FF-out" }),
        ],
    })
    return {
        fn: (x: tf.Tensor) => ff.apply(x),
        ws: () => <tf.Variable[]>[...ff.getWeights()],
    }
}
