import * as tf from "@tensorflow/tfjs"
import { MHA, positionalEncoding } from "../cnn_dddqn/model/mha"
import { AED } from "../cnn_dddqn/model/ae"

let len = 10
let d_model = 256
let head = 4
// positionalEncoding(10, 16).print()

// let { fn: mha1 } = MHA(d_model, head, d_model / head, d_model / head)
// let { fn: mha2 } = MHA(d_model, head, d_model / head, d_model / head)

// let pe = positionalEncoding(len, d_model)
// let qkv = tf.fill([len, d_model], 1)
// let mhafn = (mha, qkv) => {
//     return mha(qkv, qkv)
// }

const t = tf.tensor2d([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
])

const positionPooling = (
    x: tf.Tensor4D,
    filterSize: number | [number, number],
    strides: number | [number, number],
    pad: number | "valid" | "same"
) =>
    tf.tidy(() => {
        let [batch, h, w, c] = x.shape
        let peH = positionalEncoding(h, c).expandDims(2).tile([1, 1, w, 1])
        let peW = positionalEncoding(w, c).expandDims(1).tile([1, h, 1, 1])
        let pe = tf.add(peH, peW)
        const { result, indexes } = tf.maxPoolWithArgmax(x, filterSize, strides, pad)
        return tf.add(result, pe.flatten().gather(indexes.cast("int32")).reshapeAs(result))
    })

positionPooling(t.expandDims(0).expandDims(-1), 2, 2, "same").print()
tf.time(() => positionPooling(t.expandDims(0).expandDims(-1), 2, 2, "same")).then((time) =>
    console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`)
)

tf.time(() => positionPooling(t.expandDims(0).expandDims(-1), 2, 2, "same")).then((time) =>
    console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`)
)
