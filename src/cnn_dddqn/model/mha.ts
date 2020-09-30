import * as tf from "@tensorflow/tfjs"

const MHA = (d_model: number, h: number, d_k: number, d_v: number,) => {

    // const linears = new Array(h)
    //     .fill(0)
    //     .map(() => tf.layers.dense({ inputShape: [d_model], units: (d_k + d_k + d_v)*h }))

    const Qlinear = tf.layers.dense({ inputShape: [d_model], units: d_k * h })
    const KVlinear = tf.layers.dense({ inputShape: [d_model], units: (d_k + d_v) * h })

    const out = tf.layers.dense({ inputShape: [d_v * h], units: d_model })

    return {
        mha: (Qin: tf.Tensor, KVin: tf.Tensor) => {
            const mask = tf
                .linalg
                .bandPart(
                    tf.fill([
                        Qin.shape.slice(-2)[0],
                        KVin.shape.slice(-2)[0],
                    ], -1e9),
                    -1, 0,
                )


            const Q = (<tf.Tensor>Qlinear
                .apply(Qin))
                .split(h, -1)

            const [K, V] = (<tf.Tensor>KVlinear
                .apply(KVin))
                .split([d_k * h, d_v * h], -1)
                .map(t => t.split(h, -1))

            const att = tf
                .matMul(Q[0], K[0], false, true)
                .div(d_k ** 0.5)
                .add(mask)
                .softmax(-1)

        }
    }
}
