import * as tf from "@tensorflow/tfjs"
export declare function buildModel(
    args: {
        sequenceLen: Number,
        inputNum: Number,
        embInner: Number[],
        filters: Number,
        outputInner: Number[],
        outputNum: Number
    }
): cnnNLP

declare class cnnNLP {
    constructor(args: {
        sequenceLen: Number,
        inputNum: Number,
        embInner: Number[],
        filters: Number,
        outputInner: Number[],
        outputNum: Number
    })
    model: tf.LayersModel
    predict(
        x: tf.Tensor | tf.Tensor[],
        args?: {
            batchSize?: number
            verbose?: boolean
        }
    ): tf.Tensor | tf.Tensor[]
}