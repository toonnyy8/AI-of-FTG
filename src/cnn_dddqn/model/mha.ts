import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"

const getAngles = (pos: tf.Tensor, i: tf.Tensor, d_model: number) => tf.tidy(() => {
    const angleRates = tf.div(
        1,
        tf.pow(
            10000,
            tf.mul(
                tf.floorDiv(i, 2),
                2 / d_model,
            ),
        ),
    )
    return pos.matMul(angleRates)
})

export const positionalEncoding = (position: number, d_model: number) => {
    if (d_model % 2 == 0) {
        const angleRads = getAngles(
            tf.linspace(0, position - 1, position).expandDims(1),
            tf.linspace(0, d_model - 1, d_model).expandDims(0),
            d_model,
        )
        const [even, odd,] = angleRads.reshape([position, -1, 2]).unstack(-1)

        return tf.stack(
            [
                tf.sin(even),
                tf.cos(odd)
            ],
            -1,
        ).reshape([1, position, -1])
    }
    else {
        const angleRads = getAngles(
            tf.linspace(0, position - 1, position).expandDims(1),
            tf.linspace(0, d_model, d_model + 1).expandDims(0),
            d_model,
        )
        const [even, odd,] = angleRads.reshape([position, -1, 2]).unstack(-1)

        return tf.stack(
            [
                tf.sin(even),
                tf.cos(odd)
            ],
            -1,
        )
            .reshape([1, position, -1])
            .slice([0, 0, 0], [-1, -1, d_model])
    }
}


export const MHA = (d_model: number, h: number, d_k: number, d_v: number,) => {

    const QLinear = tf.layers.dense({ inputShape: [d_model], units: d_k * h })
    const KVLinear = tf.layers.dense({ inputShape: [d_model], units: (d_k + d_v) * h })

    const outLinear = tf.layers.dense({ inputShape: [d_v * h], units: d_model })

    return {
        fn: (Qin: tf.Tensor, KVin: tf.Tensor) => tf.tidy(() => {
            const mask = tf
                .linalg
                .bandPart(
                    tf.fill([
                        Qin.shape.slice(-2)[0],
                        KVin.shape.slice(-2)[0],
                    ], -1e9),
                    -1, 0,
                )


            let reshape2Head = (t: tf.Tensor) => {
                const shape = t.shape
                return t.reshape([h, ...shape.slice(0, -1), -1])
            }
            const Q = reshape2Head(<tf.Tensor>QLinear.apply(Qin))

            const [K, V] = (<tf.Tensor>KVLinear
                .apply(KVin))
                .split([d_k * h, d_v * h], -1)
                .map(t => reshape2Head(t))

            const att = tf
                .matMul(Q, K, false, true)
                .div(d_k ** 0.5)
                .add(mask)
                .softmax(-1)

            let headConcat = (t: tf.Tensor) => {
                const shape = t.shape
                return t.reshape([...shape.slice(1, -1), -1])
            }

            return <tf.Tensor>outLinear.apply(headConcat(tf.matMul(att, V)))
        }),
        ws: () => [
            ...QLinear.getWeights(),
            ...KVLinear.getWeights(),
            ...outLinear.getWeights(),
        ]
    }
}

export const FF = (d_model: number, hiddens: number) => {
    const linear1 = tf.layers.dense({ inputShape: [d_model], units: hiddens })
    const linear2 = tf.layers.dense({ inputShape: [hiddens], units: d_model })
    nn.mish
    return {
        fn: (x: tf.Tensor) =>
            linear2.apply(nn.mish(<tf.Tensor>linear1.apply(x))),
        ws: () => [
            ...linear1.getWeights(),
            ...linear2.getWeights(),
        ],
    }
}