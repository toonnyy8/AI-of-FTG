import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../../lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

class cnnNLP {
    constructor(
        {
            sequenceLen,
            inputNum,
            embInner = [64, 64, 64],
            filters,
            outputInner = [512, 512, 512],
            outputNum = 36
        }
    ) {
        let input = tf.input({ shape: [sequenceLen, inputNum] })
        let embLayer = tf.layers.dense({ units: embInner[0], activation: 'selu' }).apply(input)
        for (let i = 1; i < embInner.length; i++) {
            embLayer = tf.layers.dense({ units: embInner[i], activation: 'selu' }).apply(embLayer)
        }
        embLayer = tf.layers.reshape({ targetShape: [sequenceLen, embInner[embInner.length - 1], 1] }).apply(embLayer)

        let cnnLayers = new Array(sequenceLen).fill(0).map((val, idx) => {
            return tf.layers.flatten().apply(
                tf.layers.maxPooling2d({
                    poolSize: [sequenceLen - idx, 1],
                    strides: [sequenceLen - idx, 1]
                }).apply(
                    tf.layers.conv2d({
                        filters: filters,
                        kernelSize: [idx + 1, embInner[embInner.length - 1]],
                        activation: "selu"
                    }).apply(embLayer)
                )
            )
        })
        let concatLayer = tf.layers.concatenate().apply(cnnLayers)
        let outputLayer = tf.layers.dense({ units: outputInner[0], activation: 'selu' }).apply(concatLayer)
        for (let i = 1; i < outputInner.length; i++) {
            outputLayer = tf.layers.dense({ units: outputInner[i], activation: 'selu' }).apply(outputLayer)
        }
        let output = tf.layers.dense({ units: outputNum, activation: 'softmax' }).apply(outputLayer)

        this.model = tf.model({ inputs: [input], outputs: output })

        this.optimizer = tf.train.adam(1e-4)
    }

    predict(...args) {
        return this.model.predict(...args)
    }

    train() {
        this.optimizer.minimize(() => {
            return
        }, true, this.model.getWeights())
    }
}

export function buildModel(
    {
        sequenceLen,
        inputNum,
        embInner = [64, 64, 64],
        filters,
        outputInner = [512, 512, 512],
        outputNum = 36
    }
) {
    return new cnnNLP({
        sequenceLen: sequenceLen,
        inputNum: inputNum,
        embInner: embInner,
        filters: filters,
        outputInner: outputInner,
        outputNum: outputNum
    })
}