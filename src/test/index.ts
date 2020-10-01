import * as tf from "@tensorflow/tfjs"
import { MHA, positionalEncoding } from "../cnn_dddqn/model/mha"
import { AED } from "../cnn_dddqn/model/ae"

let len = 10
let d_model = 256
let head = 4
// positionalEncoding(10, 16).print()

let { fn: mha1 } = MHA(d_model, head, d_model / head, d_model / head)
let { fn: mha2 } = MHA(d_model, head, d_model / head, d_model / head)

let pe = positionalEncoding(len, d_model)
let qkv = tf.fill([len, d_model], 1)
let mhafn = (mha, qkv) => {
    return mha(
        qkv,
        qkv,
    )
}

tf.time(() => mhafn(mha2, mhafn(mha1, qkv.add(pe))))
    .then(time => console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))

tf.time(() => mhafn(mha2, mhafn(mha1, qkv.add(pe))))
    .then(time => console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))


const [{ fn: ae }, { fn: ad }] = AED(6, 3)

tf.time(() => ad(<tf.Tensor<tf.Rank.R4>>ae(<tf.Tensor<tf.Rank.R4>>tf.fill([10, 64, 64, 3], 1))))
    .then(time => console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))

tf.time(() => ad(<tf.Tensor<tf.Rank.R4>>ae(<tf.Tensor<tf.Rank.R4>>tf.fill([10, 64, 64, 3], 1))))
    .then(time => console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))
console.log(
    ae(<tf.Tensor<tf.Rank.R4>>tf.fill([10, 64, 64, 3], 1)).shape
)
