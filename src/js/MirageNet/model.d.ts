import * as tf from "@tensorflow/tfjs"
import { Transformer_XL } from "./transformer_xl"

declare class MirageNet {
    actionModel: tf.Model[]
    predictionNet: Transformer_XL
    constructor({ }: { predictionNum: number, stepNum: number })
    buildActionModel(): tf.Model
    inference(): tf.Tensor
    loss(): tf.Tensor
    train(): void
}