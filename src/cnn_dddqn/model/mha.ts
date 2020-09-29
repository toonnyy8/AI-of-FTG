import * as tf from "@tensorflow/tfjs"

const MHA = (d_model: number, h: number, d_k: number, d_v: number,) => {

    // const linears = new Array(h)
    //     .fill(0)
    //     .map(() => tf.layers.dense({ inputShape: [d_model], units: (d_k + d_k + d_v)*h }))

    const Qlinear = tf.layers.dense({ inputShape: [d_model], units: (d_k) * h })
    const KVlinear = tf.layers.dense({ inputShape: [d_model], units: (d_k + d_v) * h })

    const out = tf.layers.dense({ inputShape: [d_v * h], units: d_model })

    return {
        mha: (Qin: tf.Tensor, KVin: tf.Tensor) => {
            (<tf.Tensor>Qlinear.apply(Qin))

            KVlinear.apply(Qin)
        }
    }
}
