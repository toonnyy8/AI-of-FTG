import * as tf from "@tensorflow/tfjs"
import { TransformerXL } from "./transformerXL"

declare class MirageNet {
    actionModel: tf.LayersModel
    predictionNet: TransformerXL
    constructor({ }: { predictionNum: number, stepNum: number })
    buildActionModel(): tf.LayersModel
    inference(): tf.Tensor
    loss(): tf.Tensor
    train(): void
}