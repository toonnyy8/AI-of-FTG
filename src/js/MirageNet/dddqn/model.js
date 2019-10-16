import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"


export class DDDQN {
    constructor({
        sequenceLen = 60,
        inputNum = 10,
        embInner = [64, 64, 64],
        filters = 64,
        outputInner = [512, 512, 512],
        actionNum = 36,
        memorySize = 100,
        updateTargetStep = 20
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.count = 0

            this.actionNum = actionNum
        }

        {
            this.model = this.buildModel({
                sequenceLen: sequenceLen,
                inputNum: inputNum,
                embInner: embInner,
                filters: filters,
                outputInner: outputInner,
                actionNum: actionNum
            })

            this.targetModel = this.buildModel({
                sequenceLen: sequenceLen,
                inputNum: inputNum,
                embInner: embInner,
                filters: filters,
                outputInner: outputInner,
                actionNum: actionNum
            })

            this.targetModel.setWeights(this.model.getWeights())
        }

        {
            this.memorySize = memorySize
            this.memory = []
        }

        {
            this.optimizer = tf.train.adam(5e-4)
        }

    }

    buildModel(
        {
            sequenceLen,
            inputNum,
            embInner = [64, 64, 64],
            filters,
            outputInner = [512, 512, 512],
            actionNum = 36
        }
    ) {
        let input = tf.input({ shape: [sequenceLen, inputNum] })
        let embLayer = tf.layers.dense({ units: embInner[0], activation: 'selu' }).apply(input)
        for (let i = 1; i < embInner.length; i++) {
            embLayer = tf.layers.dense({ units: embInner[i], activation: 'selu' }).apply(embLayer)
        }
        embLayer = tf.layers.reshape({ targetShape: [sequenceLen, embInner[embInner.length - 1], 1] }).apply(embLayer)
        embLayer = tf.layers.dropout({ rate: 0.1 }).apply(embLayer)

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
        concatLayer = tf.layers.dropout({ rate: 0.1 }).apply(concatLayer)

        let outputLayer = tf.layers.dense({ units: outputInner[0], activation: 'selu' }).apply(concatLayer)
        for (let i = 1; i < outputInner.length; i++) {
            outputLayer = tf.layers.dense({ units: outputInner[i], activation: 'selu' }).apply(outputLayer)
        }
        outputLayer = tf.layers.dense({ units: actionNum, activation: 'selu' }).apply(outputLayer)
        outputLayer = tf.layers.dropout({ rate: 0.1 }).apply(outputLayer)

        let value = tf.layers.dense({
            units: actionNum,
            activation: "selu"
        }).apply(outputLayer)

        let A = tf.layers.dense({
            units: actionNum,
            activation: "selu"
        }).apply(outputLayer)

        let mean = tfex.layers(tf).lambda({
            func: (x) => {
                return tf.mean(x, 1, true)
            },
            outputShape: [1]
        }).apply([A])

        let advantage = tfex.layers(tf).lambda({
            func: (x, y) => {
                return tf.sub(x, y)
            }
        }).apply([A, mean])
        console.log(advantage)

        let Q = tf.layers.add().apply([value, advantage])

        let output = tf.layers.softmax().apply(Q)

        return tf.model({ inputs: [input], outputs: output })
    }

    loss(arrayPrevS, arrayA, arrayR, arrayNextS) {
        console.log(arrayPrevS)
        let batchPrevS = tf.tensor3d(arrayPrevS);
        let batchA = tf.tensor1d(arrayA, 'int32');
        let batchR = tf.tensor1d(arrayR);
        let batchNextS
        batchNextS = tf.tensor3d(arrayNextS);

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
            console.log(data)
            arrayPrevS.push(data[0])
            arrayA.push(data[1])
            arrayR.push(data[2])
            arrayNextS.push(data[3])
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
    sequenceLen = 60,
    inputNum = 10,
    embInner = [64, 64, 64],
    filters = 64,
    outputInner = [512, 512, 512],
    actionNum = 36,
    memorySize = 100,
    updateTargetStep = 20
}) {
    return new DDDQN({
        sequenceLen,
        inputNum,
        embInner,
        filters,
        outputInner,
        actionNum,
        memorySize,
        updateTargetStep
    })
}