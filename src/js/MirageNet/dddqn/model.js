import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../../lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

export class DDDQN {
    constructor({
        sequenceLen = 60,
        stateVectorLen = 10,
        embInner = [32, 32, 32],
        layerNum = 8,
        outputInner = [32, 32],
        actionsNum = [3, 3, 4],
        memorySize = 1000,
        updateTargetStep = 0.05,
        minLearningRate = 1e-5
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.count = 0

            this.actionsNum = actionsNum
        }

        {
            this.model = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                embInner: embInner,
                layerNum: layerNum,
                outputInner: outputInner,
                actionsNum: actionsNum
            })
            this.model.summary()

            this.targetModel = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                embInner: embInner,
                layerNum: layerNum,
                outputInner: outputInner,
                actionsNum: actionsNum
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
            actionsNum = [3, 3, 4]
        }
    ) {
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
            stateSeqLayer = stateSeqNet(stateSeqLayer, stateVectorLen + actionsNum.reduce((a, b) => a + b, 0), sequenceLen)
        }

        let outputs = actionsNum.map(actionNum => {
            let value = stateSeqLayer
            {
                value = stateSeqNet(value, stateVectorLen + actionNum, sequenceLen)
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

                A = tfex.layers.lambda({
                    func: (x) => {
                        return tf.sub(x, tf.mean(x, 1, true))
                    }
                }).apply([A])
            }

            let Q = tf.layers.add().apply([value, A])

            return Q
        })

        return tf.model({ inputs: [input], outputs: outputs })
    }

    tQandQ(batchPrevS, batchAs, batchRs, batchNextS) {
        return tf.tidy(() => {
            let predictions = this.model.predict(batchPrevS)
            const Qs = this.actionsNum.map((actionNum, actionType) => {
                return tf.mul(
                    tf.oneHot(
                        batchAs[actionType],
                        actionNum
                    ),
                    predictions[actionType]
                ).sum(1)
            })

            let targetPredictions = this.targetModel.predict(batchNextS)
            const targetQs = this.actionsNum.map((actionNum, actionType) => {
                const maxQ = tf.mul(
                    tf.oneHot(
                        tf.argMax(
                            predictions[actionType],
                            1
                        ),
                        actionNum
                    ),
                    targetPredictions[actionType]
                ).sum(1)
                const targets = batchRs[actionType].add(maxQ.mul(tf.scalar(0.99)));
                return targets;
            })
            return [targetQs, Qs]
        })
    }

    train(replayNum = 100, loadIdxes = [null], usePrioritizedReplay = false) {
        tf.tidy(() => {
            let train_ = (replayIdxes) => {
                tf.tidy(() => {
                    let replayIdxes_ = replayIdxes.slice()

                    let arrayPrevS = []
                    let arrayAs = new Array(this.actionsNum.length).fill([])
                    let arrayRs = new Array(this.actionsNum.length).fill([])
                    let arrayNextS = []

                    for (let i = 0; i < replayNum; i++) {
                        if (replayIdxes_[i] == null || replayIdxes_[i] >= this.memory.length) {
                            replayIdxes_[i] = Math.floor(Math.random() * this.memory.length);
                        }
                        let data = this.memory[replayIdxes_[i]]
                        // console.log(data)
                        arrayPrevS.push(data.prevS)
                        for (let j = 0; j < this.actionsNum.length; j++) {
                            arrayAs[j][i] = data.As[j]
                            arrayRs[j][i] = data.Rs[j]
                        }
                        arrayNextS.push(data.nextS)
                    }

                    let batchPrevS = tf.tensor3d(arrayPrevS)
                    let batchAs = arrayAs.map((arrayA) => {
                        return tf.tensor1d(arrayA, 'int32')
                    })
                    let batchRs = arrayRs.map((arrayR) => {
                        return tf.tensor1d(arrayR, 'int32')
                    })
                    let batchNextS = tf.tensor3d(arrayNextS)

                    let grads = this.optimizer.computeGradients(
                        () => {
                            let [targetQs, Qs] = this.tQandQ(
                                batchPrevS,
                                batchAs,
                                batchRs,
                                batchNextS
                            )
                            tf.addN(
                                this.actionsNum.map((actionNum, actionType) => {
                                    return tf.abs(tf.sub(targetQs[actionType], Qs[actionType]))
                                })
                            ).arraySync()
                                .forEach((absTD, idx) => {
                                    this.memory[replayIdxes_[idx]].p = absTD
                                })
                            let loss = tf.mean(
                                tf.stack(
                                    this.actionsNum.map((actionNum, actionType) => {
                                        return tf.losses.huberLoss(targetQs[actionType], Qs[actionType])
                                    })
                                )
                            )
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
                        let prioritys = tf.tensor(this.memory.map(mem => mem.p))
                        prioritys = tf.softmax(prioritys)
                        // prioritys.print()
                        return tf.multinomial(prioritys, replayNum, null, true).arraySync()
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

    store(preState, actions, rewards, nextState) {
        if (this.memory.length == this.memorySize) {
            this.memory.pop()
        }
        this.memory.unshift({
            prevS: preState,
            As: actions,
            Rs: rewards,
            nextS: nextState,
            p: 1e+9
        })
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
    actionsNum = [3, 3, 4],
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
        actionsNum,
        memorySize,
        updateTargetStep,
        minLearningRate
    })
}