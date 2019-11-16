import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../../lib/tfjs-extensions/src"
import { transformerXL } from "../transformerXL_sp"
const tfex = registerTfex(tf)

export class DDDQN {
    constructor({
        sequenceLen = 60,
        stateVectorLen = 10,
        embInner = [32, 32, 32],
        layerNum = 8,
        outputInner = [32, 32],
        actionNum = 8,
        memorySize = 1000,
        updateTargetStep = 0.05,
        minLearningRate = 1e-5
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.count = 0

            this.actionNum = actionNum
        }

        {
            this.model = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                embInner: embInner,
                layerNum: layerNum,
                outputInner: outputInner,
                actionNum: actionNum
            })
            this.model.summary()

            this.targetModel = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                embInner: embInner,
                layerNum: layerNum,
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
            this.minLearningRate = minLearningRate
            this.optimizer = tf.train.adam(1e-3)
        }

    }

    buildModel(
        {
            sequenceLen,
            stateVectorLen,
            layerNum = 32,
            actionNum = 9
        }
    ) {
        class model {
            constructor(
                args = {

                },
                scope = tfex.scope
            ) {
                this.scope = scope

            }
            predict() {

            }
        }
        let stateSeqNet = (inputLayer, stateVectorLen, sequenceLen) => {
            stateSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen,
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(inputLayer)
            stateSeqLayer = tf.layers.batchNormalization({}).apply(stateSeqLayer)

            stateSeqLayer = tf.layers.permute({
                dims: [2, 1]
            }).apply(stateSeqLayer)

            stateSeqLayer = tf.layers.conv1d({
                filters: sequenceLen,
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
            stateSeqLayer = tf.layers.batchNormalization({}).apply(stateSeqLayer)

            stateSeqLayer = tf.layers.permute({
                dims: [2, 1]
            }).apply(stateSeqLayer)

            return stateSeqLayer
        }
        let input = tf.input({ shape: [sequenceLen, stateVectorLen] })

        let stateSeqLayer = input

        for (let i = 0; i < layerNum; i++) {
            stateSeqLayer = stateSeqNet(stateSeqLayer, stateVectorLen + actionNum, sequenceLen)
        }

        let value = stateSeqLayer
        {
            value = stateSeqNet(value, stateVectorLen + actionNum, sequenceLen)

            //用Global Average Pooling代替Fully Connected
            value = tf.layers.globalAveragePooling1d({}).apply(value)
            value = tf.layers.reshape({ targetShape: [1, stateVectorLen + actionNum] }).apply(value)

            value = tf.layers.conv1d({
                filters: 1,
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(value)
            value = tf.layers.flatten().apply(value)
        }

        let A = stateSeqLayer
        {
            A = stateSeqNet(A, stateVectorLen + actionNum, sequenceLen)

            //用Global Average Pooling代替Fully Connected
            A = tf.layers.globalAveragePooling1d({}).apply(A)
            A = tf.layers.reshape({ targetShape: [1, stateVectorLen + actionNum] }).apply(A)

            A = tf.layers.conv1d({
                filters: actionNum,
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(A)
            A = tf.layers.flatten().apply(A)
        }

        let advantage = tfex.layers.lambda({
            func: (x) => {
                return tf.sub(x, tf.mean(x, 1, true))
            }
        }).apply([A])

        let Q = tf.layers.add().apply([value, advantage])

        return tf.model({ inputs: [input], outputs: Q })
    }

    loss(arrayPrevS, arrayA, arrayR, arrayNextS) {
        let calcTarget = (batchR, batchNextS) => {
            return tf.tidy(() => {
                const maxQ = tf.mul(
                    tf.oneHot(
                        tf.argMax(
                            this.model.predict(batchNextS),
                            1
                        ),
                        this.actionNum
                    ),
                    this.targetModel.predict(batchNextS)
                ).sum(1)
                const targets = batchR.add(maxQ.mul(tf.scalar(0.99)));
                return targets;
            });
        }
        return tf.tidy(() => {
            // console.log(arrayPrevS)
            let batchPrevS = tf.tensor3d(arrayPrevS)
            let batchA = tf.tensor1d(arrayA, 'int32')
            let batchR = tf.tensor1d(arrayR)
            let batchNextS = tf.tensor3d(arrayNextS)

            const Qs = tf.tidy(() => {
                return tf.mul(
                    tf.oneHot(batchA, this.actionNum),
                    this.model.predict(batchPrevS)
                ).sum(1)
            })

            const targetQs = calcTarget(batchR, batchNextS)

            return tf.mean(
                tf.square(
                    tf.sub(
                        Qs,
                        targetQs
                    )
                )
            )
        })

    }

    train(replayNum = 100, loadIdxes = [null], usePrioritizedReplay = false) {
        tf.tidy(() => {
            let train_ = (replayIdxes) => {
                tf.tidy(() => {
                    let arrayPrevS = []
                    let arrayA = []
                    let arrayR = []
                    let arrayNextS = []

                    for (let i = 0; i < replayNum; i++) {
                        let data = this.load(replayIdxes[i])
                        // console.log(data)
                        arrayPrevS.push(data[0])
                        arrayA.push(data[1])
                        arrayR.push(data[2])
                        arrayNextS.push(data[3])
                    }

                    let grads = this.optimizer.computeGradients(
                        () => {
                            let loss = this.loss(arrayPrevS, arrayA, arrayR, arrayNextS)
                            loss.print()
                            return loss
                        }, this.model.getWeights(true)).grads

                    let gradsName = Object.keys(grads)
                    grads = tfex.funcs.clipByGlobalNorm(Object.values(grads), 0.05)[0]

                    this.optimizer.applyGradients(gradsName.reduce((acc, gn, idx) => {
                        acc[gn] = grads[idx]
                        // if (gn == "weighted_average_WeightedAverage1/w") {
                        //     acc[gn].print()
                        // }
                        return acc
                    }, {}))

                    this.count++

                    this.optimizer.learningRate = (1e-4 / this.count ** 0.5) + this.minLearningRate

                    this.targetModel.setWeights(
                        this.targetModel.getWeights().map((weight, idx) => {
                            return tf.add(
                                tf.mul(this.model.getWeights()[idx], this.updateTargetStep),
                                tf.mul(weight, 1 - this.updateTargetStep),
                            )
                        })
                    )
                })
            }
            if (this.memory.length != 0) {
                if (usePrioritizedReplay) {
                    let prioritizedReplayBuffer = tf.tidy(() => {
                        let e = tf.tensor(this.memory.map(mem => mem[2]))
                        e = tf.abs(e.sub(e.mean()))
                        e = e.div(e.sum(0, true))
                        // e.print()
                        return tf.multinomial(e, replayNum, null, true).arraySync()
                    })
                    // console.log(prioritizedReplayBuffer)
                    train_(prioritizedReplayBuffer.map((prioritizedReplayIdx, idx) => {
                        return loadIdxes[idx] == null || loadIdxes[idx] == undefined ? prioritizedReplayIdx : loadIdxes[idx]
                    }))
                } else {
                    train_(loadIdxes)
                }
            }
        })
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
    stateVectorLen = 10,
    embInner = [32, 32, 32],
    layerNum = 8,
    outputInner = [32, 32],
    actionNum = 8,
    memorySize = 1000,
    updateTargetStep = 0.05,
    minLearningRate = 1e-3
}) {
    return new DDDQN({
        sequenceLen,
        stateVectorLen,
        embInner,
        layerNum,
        outputInner,
        actionNum,
        memorySize,
        updateTargetStep,
        minLearningRate
    })
}