import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../../lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

export class DDDQN {
    constructor({
        sequenceLen = 4,
        stateVectorLen = 59,
        layerNum = 8,
        actionsNum = [2, 2, 2, 2, 2, 2, 2],
        memorySize = 1000,
        updateTargetStep = 0.05,
        initLearningRate = 1e-3,
        minLearningRate = 1e-5,
        discounts = [1, 1, 1, 1, 0.1, 0.1, 0.1]
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.discounts = discounts

            this.count = 0

            this.actionsNum = actionsNum
        }

        {
            this.model = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                layerNum: layerNum,
                actionsNum: actionsNum
            })
            this.model.summary()

            this.targetModel = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                layerNum: layerNum,
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
            this.initLearningRate = initLearningRate
            this.optimizer = tf.train.adam(this.initLearningRate)
        }

    }

    buildModel({
        sequenceLen,
        stateVectorLen,
        layerNum = 32,
        actionsNum = [3, 3, 4]
    }) {
        let input = tf.input({ shape: [sequenceLen, stateVectorLen] })
        let stateSeqLayer = tfex.layers.lambda({
            func: (x) => {
                let [player1, player2] = tf.split(x, 2, 2)
                return tf.concat([x, tf.concat([player2, player1], 2)], 0)
            },
            outputShape: [null, sequenceLen, stateVectorLen]
        }).apply([input])

        for (let i = 0; i < layerNum; i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen * 2,
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        stateSeqLayer = tf.layers.conv1d({
            filters: stateVectorLen * 2,
            kernelSize: [sequenceLen],
            activation: "selu",
        }).apply(stateSeqLayer)

        let value = stateSeqLayer
        {
            value = tf.layers.conv1d({
                filters: 1,
                kernelSize: [1],
                padding: "same",
                activation: "selu"
            }).apply(value)
            value = tf.layers.flatten().apply(value)
        }

        let A = stateSeqLayer

        {
            A = tf.layers.conv1d({
                filters: actionsNum.reduce((prev, curr) => prev + curr, 0),
                kernelSize: [1],
                padding: "same",
                activation: "selu"
            }).apply(A)
            A = tf.layers.flatten().apply(A)

            A = tfex.layers.lambda({
                func: (x) => {
                    return tf.sub(x, tf.mean(x, 1, true))
                }
            }).apply([A])
        }

        let Q = tf.layers.add().apply([value, A])

        let [player1Q, player2Q] = tfex.layers.lambda({
            func: (x) => {
                let [player1Q, player2Q] = tf.split(x, 2, 0)
                return [player1Q, tfex.funcs.stopGradient(player2Q)]
            },
            outputShape: [
                [null, actionsNum.reduce((prev, curr) => prev + curr, 0)],
                [null, actionsNum.reduce((prev, curr) => prev + curr, 0)]
            ]
        }).apply([Q])

        Q = player1Q

        player1Q = tf.layers.repeatVector({ n: actionsNum.reduce((prev, curr) => prev + curr, 0), inputShape: [actionsNum.reduce((prev, curr) => prev + curr, 0)] }).apply(player1Q)
        player1Q = tf.layers.permute({
            dims: [2, 1],
            inputShape: [actionsNum.reduce((prev, curr) => prev + curr, 0), actionsNum.reduce((prev, curr) => prev + curr, 0)]
        }).apply(player1Q)

        let opponentAdvantage = tf.layers.multiply({}).apply([
            player1Q, player2Q
        ])// 對手優勢分析
        let opponentDisadvantage = tf.layers.multiply({}).apply([
            player1Q,
            tfex.layers.lambda({
                func: (x) => { return tf.mul(x, -1) }
            }).apply([player2Q])
        ])// 對手劣勢分析

        opponentAdvantage = tf.layers.reshape({
            targetShape: [actionsNum.reduce((prev, curr) => prev + curr, 0), actionsNum.reduce((prev, curr) => prev + curr, 0), 1]
        }).apply(opponentAdvantage)

        opponentDisadvantage = tf.layers.reshape({
            targetShape: [actionsNum.reduce((prev, curr) => prev + curr, 0), actionsNum.reduce((prev, curr) => prev + curr, 0), 1]
        }).apply(opponentDisadvantage)

        let adversarial = tfex.layers.lambda({
            func: (x, y) => {
                return tf.concat([x, y], 3)
            },
            outputShape: [null, actionsNum.reduce((prev, curr) => prev + curr, 0), actionsNum.reduce((prev, curr) => prev + curr, 0), 2]
        }).apply([opponentAdvantage, opponentDisadvantage])

        adversarial = tf.layers.conv2d({
            filters: actionsNum.reduce((prev, curr) => prev + curr, 0) * 2,
            kernelSize: actionsNum.reduce((prev, curr) => prev + curr, 0),
            activation: "selu"
        }).apply(adversarial)
        adversarial = tf.layers.conv2d({
            filters: actionsNum.reduce((prev, curr) => prev + curr, 0),
            kernelSize: 1,
            activation: "selu"
        }).apply(adversarial)

        adversarial = tf.layers.flatten().apply(adversarial)

        //    outputs = tf.layers.add({}).apply([Q, adversarial])
        let outputs = tfex.layers.lambda({
            func: (outputLayer) => {
                return tf.split(outputLayer, actionsNum, 1)
            },
            outputShape: actionsNum.map(actionNum => [null, actionNum])
        }).apply([adversarial])


        return tf.model({ inputs: [input], outputs: outputs })
    }

    tQandQ(batchPrevS, batchAs, batchRs, batchNextS) {
        return tf.tidy(() => {
            let predictions = this.model.predictOnBatch(batchPrevS)
            if (this.actionsNum.length == 1) {
                predictions = [predictions]
            }
            const Qs = this.actionsNum.map((actionNum, actionType) => {
                return tf.mul(
                    tf.oneHot(
                        batchAs[actionType],
                        actionNum
                    ),
                    predictions[actionType]
                ).sum(1)
            })

            let targetPredictions = this.targetModel.predictOnBatch(batchNextS)
            if (this.actionsNum.length == 1) {
                targetPredictions = [targetPredictions]
            }
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
                const targets = batchRs[actionType].add(maxQ.mul(tf.scalar(this.discounts[actionType])));
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
                        return tf.tensor1d(arrayR, 'float32')
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
                            // tf.addN(
                            //     this.actionsNum.map((actionNum, actionType) => {
                            //         return tf.abs(tf.sub(targetQs[actionType], Qs[actionType]))
                            //     })
                            // ).arraySync()
                            //     .forEach((absTD, idx) => {
                            //         this.memory[replayIdxes_[idx]].p = absTD
                            //     })
                            let loss = tf.mean(
                                tf.stack(
                                    this.actionsNum.map((actionNum, actionType) => {
                                        // return tf.losses.huberLoss(targetQs[actionType], Qs[actionType])
                                        return tf.losses.meanSquaredError(targetQs[actionType], Qs[actionType])
                                    })
                                )
                            )
                            loss.print()
                            return loss
                        }, this.model.getWeights(true)).grads

                    let gradsName = Object.keys(grads)
                    grads = tfex.funcs.clipByGlobalNorm(Object.values(grads), 1)[0]

                    this.optimizer.applyGradients(gradsName.reduce((acc, gn, idx) => {
                        acc[gn] = grads[idx]
                        // if (gn == "weighted_average_WeightedAverage1/w") {
                        //     acc[gn].print()
                        // }
                        return acc
                    }, {}))

                    this.count++

                    this.optimizer.learningRate = Math.max(this.initLearningRate / (this.count ** 0.5), this.minLearningRate)

                    if (this.updateTargetStep < 1) {
                        this.targetModel.setWeights(
                            this.targetModel.getWeights().map((weight, idx) => {
                                return tf.add(
                                    tf.mul(this.model.getWeights()[idx], this.updateTargetStep),
                                    tf.mul(weight, 1 - this.updateTargetStep),
                                )
                            })
                        )
                    } else {
                        if (this.count % Math.round(this.updateTargetStep) == 0) {
                            this.targetModel.setWeights(this.model.getWeights())
                        }
                    }
                })
            }
            if (this.memory.length != 0) {
                if (usePrioritizedReplay) {
                    let prioritizedReplayBuffer = tf.tidy(() => {
                        let prioritys = tf.tensor(this.memory.map(mem => mem.p))
                        prioritys = tf.div(prioritys, tf.sum(prioritys, 0, true))
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

    updatePrioritys(bsz = 64) {

        if (this.memory.length != 0) {
            for (let begin = 0; begin < this.memory.length; begin += bsz) {
                tf.tidy(() => {
                    let arrayPrevS = []
                    let arrayAs = new Array(this.actionsNum.length).fill([])
                    let arrayRs = new Array(this.actionsNum.length).fill([])
                    let arrayNextS = []

                    for (let i = begin; i < Math.min(this.memory.length, begin + bsz); i++) {
                        let data = this.memory[i]
                        arrayPrevS.push(data.prevS)
                        for (let j = 0; j < this.actionsNum.length; j++) {
                            arrayAs[j][i - begin] = data.As[j]
                            arrayRs[j][i - begin] = data.Rs[j]
                        }
                        arrayNextS.push(data.nextS)
                    }

                    let batchPrevS = tf.tensor3d(arrayPrevS)
                    let batchAs = arrayAs.map((arrayA) => {
                        return tf.tensor1d(arrayA, 'int32')
                    })
                    let batchRs = arrayRs.map((arrayR) => {
                        return tf.tensor1d(arrayR, 'float32')
                    })
                    let batchNextS = tf.tensor3d(arrayNextS)

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
                            this.memory[begin + idx].p = absTD
                        })

                })
            }
        }
    }

}

export function dddqn({
    sequenceLen = 4,
    stateVectorLen = 59,
    layerNum = 8,
    actionsNum = [2, 2, 2, 2, 2, 2, 2],
    memorySize = 1000,
    updateTargetStep = 0.05,
    initLearningRate = 1e-3,
    minLearningRate = 1e-5,
    discounts = [1, 1, 1, 1, 0.1, 0.1, 0.1]
}) {
    return new DDDQN({
        sequenceLen,
        stateVectorLen,
        layerNum,
        actionsNum,
        memorySize,
        updateTargetStep,
        initLearningRate,
        minLearningRate,
        discounts
    })
}