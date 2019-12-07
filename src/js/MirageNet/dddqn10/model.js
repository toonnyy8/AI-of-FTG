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
        discount = 0.63,
        maxCoderSize = 4
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.discount = discount

            this.count = 0

            this.actionsNum = actionsNum
        }

        {
            this.model = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                layerNum: layerNum,
                actionsNum: actionsNum,
                maxCoderSize: maxCoderSize
            }, tfex.scope.variableScope("eval"))
            // this.model.summary()

            this.targetModel = this.buildModel({
                sequenceLen: sequenceLen,
                stateVectorLen: stateVectorLen,
                layerNum: layerNum,
                actionsNum: actionsNum,
                maxCoderSize: maxCoderSize
            }, tfex.scope.variableScope("target"))

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
        actionsNum = [3, 3, 4],
        maxCoderSize = 4
    }, scope = tfex.scope) {
        class m {
            constructor({
                sequenceLen,
                stateVectorLen,
                layerNum = 32,
                actionsNum = [3, 3, 4],
                maxCoderSize = 4
            }, scope = tfex.scope) {
                this.sequenceLen = sequenceLen
                this.stateVectorLen = stateVectorLen
                this.layerNum = layerNum
                this.actionsNum = actionsNum
                this.scope = scope
                this.weights = []

                let inputSize = stateVectorLen
                let outputSize = stateVectorLen * maxCoderSize
                this.weights.push(scope.getVariable(`input_w`, [1, inputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                this.weights.push(scope.getVariable(`input_b`, [1, 1, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))

                let coderNum = Math.ceil(layerNum / 4) * 2
                for (let i = 1; i <= coderNum; i++) {
                    inputSize = outputSize
                    outputSize = (stateVectorLen / 2) * (i / coderNum) + stateVectorLen * maxCoderSize * (coderNum - i) / coderNum
                    outputSize = Math.ceil(outputSize)
                    this.weights.push(scope.getVariable(`ae_w${i}`, [Math.ceil(sequenceLen / (coderNum)) + 1, inputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                    this.weights.push(scope.getVariable(`ae_b${i}`, [1, 1, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                }

                inputSize = outputSize
                this.weights.push(scope.getVariable(`ae_w${coderNum + 1}`, [Math.ceil(sequenceLen / (coderNum)) + 1, inputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                this.weights.push(scope.getVariable(`ae_b${coderNum + 1}`, [1, 1, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))

                for (let i = 1; i <= coderNum; i++) {
                    outputSize = (stateVectorLen / 2) * ((coderNum - i) / coderNum) + stateVectorLen * maxCoderSize * i / coderNum
                    outputSize = Math.ceil(outputSize)
                    this.weights.push(scope.getVariable(`ae_b${coderNum + i + 1}`, [1, 1, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                }

                {
                    this.weights.push(scope.getVariable(`value_w1`, [1, sequenceLen, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                    this.weights.push(scope.getVariable(`value_b1`, [1, 1, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))

                    this.weights.push(scope.getVariable(`value_w2`, [1, stateVectorLen * maxCoderSize, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                    this.weights.push(scope.getVariable(`value_b2`, [1, 1, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                }

                {
                    this.weights.push(scope.getVariable(`A_w1`, [1, sequenceLen, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                    this.weights.push(scope.getVariable(`A_b1`, [1, 1, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))

                    this.weights.push(scope.getVariable(`A_w2`, [1, stateVectorLen * maxCoderSize, actionsNum.reduce((prev, curr) => prev + curr, 0)], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                    this.weights.push(scope.getVariable(`A_b2`, [1, 1, actionsNum.reduce((prev, curr) => prev + curr, 0)], "float32", tf.initializers.truncatedNormal({ stddev: 0.1 })))
                }
                // this.weights.forEach(w => w.sum().print())
                console.log(this.scope.variables)
            }

            predict(x) {
                return tf.tidy(() => {
                    let stateSeqLayer = tf.conv1d(
                        x,
                        this.scope.getVariable(`input_w`),
                        1,
                        "same"
                    )
                    stateSeqLayer = tf.add(stateSeqLayer, this.scope.getVariable(`input_b`))
                    stateSeqLayer = tf.selu(stateSeqLayer)

                    let coderNum = Math.ceil(layerNum / 4) * 2
                    for (let i = 1; i <= coderNum; i++) {
                        stateSeqLayer = tf.conv1d(
                            stateSeqLayer,
                            this.scope.getVariable(`ae_w${i}`),
                            1,
                            "same"
                        )
                        stateSeqLayer = tf.add(stateSeqLayer, this.scope.getVariable(`ae_b${i}`))
                        stateSeqLayer = tf.selu(stateSeqLayer)
                    }

                    stateSeqLayer = tf.conv1d(
                        stateSeqLayer,
                        this.scope.getVariable(`ae_w${coderNum + 1}`),
                        1,
                        "same"
                    )
                    stateSeqLayer = tf.add(stateSeqLayer, this.scope.getVariable(`ae_b${coderNum + 1}`))
                    stateSeqLayer = tf.selu(stateSeqLayer)

                    for (let i = 0; i < coderNum; i++) {
                        stateSeqLayer = tf.conv1d(
                            stateSeqLayer,
                            tf.transpose(
                                this.scope.getVariable(`ae_w${coderNum - i}`),
                                [0, 2, 1]
                            ),
                            1,
                            "same"
                        )
                        stateSeqLayer = tf.add(stateSeqLayer, this.scope.getVariable(`ae_b${coderNum + i + 2}`))
                        stateSeqLayer = tf.selu(stateSeqLayer)
                    }

                    let value = stateSeqLayer
                    {
                        value = tf.transpose(value, [0, 2, 1])
                        value = tf.conv1d(
                            value,
                            this.scope.getVariable(`value_w1`),
                            1, 0
                        )
                        value = tf.add(value, this.scope.getVariable(`value_b1`))

                        value = tf.transpose(value, [0, 2, 1])
                        value = tf.conv1d(
                            value,
                            this.scope.getVariable(`value_w2`),
                            1, 0
                        )
                        value = tf.add(value, this.scope.getVariable(`value_b2`))
                        value = tf.reshape(value, [-1, 1])
                    }

                    let A = stateSeqLayer

                    {
                        A = tf.transpose(A, [0, 2, 1])
                        A = tf.conv1d(
                            A,
                            this.scope.getVariable(`A_w1`),
                            1, 0
                        )
                        A = tf.add(A, this.scope.getVariable(`A_b1`))
                        A = tf.transpose(A, [0, 2, 1])
                        A = tf.conv1d(
                            A,
                            this.scope.getVariable(`A_w2`),
                            1, 0
                        )
                        A = tf.add(A, this.scope.getVariable(`A_b2`))
                        A = tf.reshape(A, [-1, this.actionsNum.reduce((prev, curr) => prev + curr, 0)])

                        A = tf.sub(A, tf.mean(A, 1, true))
                    }

                    let Q = tf.add(value, A)

                    let outputs
                    if (this.actionsNum.length == 1) {
                        outputs = Q
                    } else {
                        outputs = tf.split(Q, this.actionsNum, 1)
                    }

                    return outputs
                })
            }

            getWeights() {
                return this.weights
            }

            setWeights(weights) {
                this.weights.forEach((w, idx) => w.assign(weights[idx]))
            }
        }

        return new m({
            sequenceLen,
            stateVectorLen,
            layerNum,
            actionsNum,
            maxCoderSize
        }, scope)
    }

    tQandQ(batchPrevS, batchAs, batchRs, batchNextS, batchDiscount) {
        return tf.tidy(() => {
            let evalNet = this.model.predict(batchPrevS)
            if (this.actionsNum.length == 1) {
                evalNet = [evalNet]
            }
            const Qs = this.actionsNum.map((actionNum, actionType) => {
                return tf.mul(
                    tf.oneHot(
                        batchAs[actionType],
                        actionNum
                    ),
                    evalNet[actionType]
                ).sum(1)
            })

            let predictions = this.model.predict(batchNextS)
            if (this.actionsNum.length == 1) {
                predictions = [predictions]
            }
            let targetPredictions = this.targetModel.predict(batchNextS)
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
                const targets = batchRs[actionType].add(maxQ.mul(batchDiscount));
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
                    let arrayDiscount = []

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
                        arrayDiscount.push(data.discount)
                    }

                    let batchPrevS = tf.tensor3d(arrayPrevS)
                    let batchAs = arrayAs.map((arrayA) => {
                        return tf.tensor1d(arrayA, 'int32')
                    })
                    let batchRs = arrayRs.map((arrayR) => {
                        return tf.tensor1d(arrayR, 'float32')
                    })
                    let batchNextS = tf.tensor3d(arrayNextS)
                    let batchDiscount = tf.tensor1d(arrayDiscount)

                    let grads = this.optimizer.computeGradients(
                        () => {
                            let [targetQs, Qs] = this.tQandQ(
                                batchPrevS,
                                batchAs,
                                batchRs,
                                batchNextS,
                                batchDiscount
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

                    // let gradsName = Object.keys(grads)
                    // grads = tfex.funcs.clipByGlobalNorm(Object.values(grads), 1)[0]

                    // this.optimizer.applyGradients(gradsName.reduce((acc, gn, idx) => {
                    //     acc[gn] = grads[idx]
                    //     // if (gn == "weighted_average_WeightedAverage1/w") {
                    //     //     acc[gn].print()
                    //     // }
                    //     return acc
                    // }, {}))
                    this.optimizer.applyGradients(grads)

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

    store(preState, actions, rewards, nextState, discount) {
        if (this.memory.length == this.memorySize) {
            this.memory.pop()
        }
        this.memory.unshift({
            prevS: preState,
            As: actions,
            Rs: rewards,
            nextS: nextState,
            discount: discount,
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
                    let arrayDiscount = []

                    for (let i = begin; i < Math.min(this.memory.length, begin + bsz); i++) {
                        let data = this.memory[i]
                        arrayPrevS.push(data.prevS)
                        for (let j = 0; j < this.actionsNum.length; j++) {
                            arrayAs[j][i - begin] = data.As[j]
                            arrayRs[j][i - begin] = data.Rs[j]
                        }
                        arrayNextS.push(data.nextS)
                        arrayDiscount.push(data.discount)
                    }

                    let batchPrevS = tf.tensor3d(arrayPrevS)
                    let batchAs = arrayAs.map((arrayA) => {
                        return tf.tensor1d(arrayA, 'int32')
                    })
                    let batchRs = arrayRs.map((arrayR) => {
                        return tf.tensor1d(arrayR, 'float32')
                    })
                    let batchNextS = tf.tensor3d(arrayNextS)
                    let batchDiscount = tf.tensor1d(arrayDiscount)

                    let [targetQs, Qs] = this.tQandQ(
                        batchPrevS,
                        batchAs,
                        batchRs,
                        batchNextS,
                        batchDiscount
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
    discount = 0.63,
    maxCoderSize = 4
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
        discount,
        maxCoderSize
    })
}