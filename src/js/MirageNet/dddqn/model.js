import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"
import * as cnnNLP from "../cnnNLP"

export class DDDQN {
    constructor({
        sequenceLen = 60,
        inputNum = 10,
        embInner = [64, 64, 64],
        filters = 64,
        outputInner = [512, 512, 512],
        outputNum = 36,
        dueling = true,
        memorySize = 100,
        updateTargetStep = 20
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.count = 0

            this.actionNum = actionNum

            this.inputShape = inputShape

            this.conv = conv
        }

        {
            this.model = this.buildModel({
                sequenceLen: sequenceLen,
                inputNum: inputNum,
                embInner: embInner,
                filters: filters,
                outputInner: outputInner,
                outputNum: outputNum
            }).model

            this.targetModel = this.buildModel({
                sequenceLen: sequenceLen,
                inputNum: inputNum,
                embInner: embInner,
                filters: filters,
                outputInner: outputInner,
                outputNum: outputNum
            }).model

            this.targetModel.setWeights(this.model.getWeights())
        }

        {
            this.memorySize = memorySize
            this.memory = []
        }

        {
            this.optimizer = tf.train.adam()
        }

    }

    buildModel(
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
        let embLayer = tf.layers.dense({ units: embInner[0], activation: 'linear' }).apply(input)
        for (let i = 1; i < embInner.length; i++) {
            embLayer = tf.layers.dense({ units: embInner[i], activation: 'linear' }).apply(embLayer)
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
                        activation: "linear"
                    }).apply(embLayer)
                )
            )
        })
        let concatLayer = tf.layers.concatenate().apply(cnnLayers)
        let outputLayer = tf.layers.dense({ units: outputInner[0], activation: 'linear' }).apply(concatLayer)
        for (let i = 1; i < outputInner.length; i++) {
            outputLayer = tf.layers.dense({ units: outputInner[i], activation: 'linear' }).apply(outputLayer)
        }
        let outputLayer = tf.layers.dense({ units: outputNum, activation: 'linear' }).apply(outputLayer)

        let value = tf.layers.dense({
            units: actionNum,
            activation: "linear"
        }).apply(outputLayer)

        let A = tf.layers.dense({
            units: actionNum,
            activation: "linear"
        }).apply(outputLayer)

        let mean = tfex.layers.lambda({
            func: (x) => {
                return tf.mean(x, 1, true)
            }
        }).apply(A)

        let advantage = tfex.layers.lambda({
            func: (x, y) => {
                return tf.sub(x, y)
            }
        }).apply([A, mean])

        Q = tf.layers.add().apply([value, advantage])

        output = tf.layers.softmax().apply(Q)

        return tf.model({ inputs: [input], outputs: output })
    }

    loss(arrayPrevS, arrayA, arrayR, arrayNextS) {
        let batchPrevS
        if (this.conv) {
            batchPrevS = tf.tensor(arrayPrevS, [arrayPrevS.length, 25, 100, 3]);
        } else {
            batchPrevS = tf.tensor2d(arrayPrevS);
        }
        let batchA = tf.tensor1d(arrayA, 'int32');
        let batchR = tf.tensor1d(arrayR);
        let batchNextS
        if (this.conv) {
            batchNextS = tf.tensor(arrayNextS, [arrayPrevS.length, 25, 100, 3]);
        } else {
            batchNextS = tf.tensor2d(arrayNextS);
        }

        const maxQ = this.targetModel.predict(batchNextS).reshape([arrayPrevS.length, this.actionNum]).max(1)

        const x = tf.variable(batchPrevS);

        batchPrevS.dispose()
        batchNextS.dispose()
        tf.nextFrame()
        return tf.tidy(() => {
            const predMask = tf.oneHot(batchA, this.actionNum);
            batchA.dispose()
            const targets = batchR.add(maxQ.mul(tf.scalar(0.99)));
            batchR.dispose()
            maxQ.dispose()
            const predictions = this.model.predict(x);
            x.dispose()
            return tf.mul(predictions.sub(targets.expandDims(1)).square(), predMask.asType('float32')).mean();
        })

    }

    async train(replayNum = 100) {
        let arrayPrevS = []
        let arrayA = []
        let arrayR = []
        let arrayNextS = []

        for (let i = 0; i < replayNum; i++) {
            let data = this.load()
            arrayPrevS.push(this.conv ? data[0].data : data[0])
            arrayA.push(data[1])
            arrayR.push(data[2])
            arrayNextS.push(this.conv ? data[3].data : data[3])
        }

        this.optimizer.minimize(() => {
            let loss = this.loss(arrayPrevS, arrayA, arrayR, arrayNextS)
            // loss.print()
            return loss
        }, true, this.model.getWeights());

        this.count++

        if (this.count >= this.updateTargetStep) {

            this.targetModel.setWeights(this.model.getWeights())
            this.count = 0
        }
    }

    store(preState, action, reward, nextState) {
        if (this.memory.length == this.memorySize) {
            this.memory.pop()
        }
        this.memory.unshift([preState, action, reward, nextState])
    }

    load(index) {
        if (index == null || index >= this.memory.length) {
            index = Math.floor(Math.random() * this.memory.length);
        }
        return this.memory[index]
    }

}

export function dddqn({
    actionNum = 2,
    inputShape = [5, 10],
    hiddenLayers = [64, 64],
    activation = "softmax",
    conv = false,
    dueling = true,
    memorySize = 100,
    updateTargetStep = 20
}) {
    return new DDDQN({
        actionNum,
        inputShape,
        hiddenLayers,
        activation,
        conv,
        dueling,
        memorySize,
        updateTargetStep
    })
}