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
        maxCoderSize = 4
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
                this.weights.push(scope.getVariable(`input_w`, [1, inputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                this.weights.push(scope.getVariable(`input_b`, [1, 1, outputSize], "float32", tf.initializers.zeros({})))

                this.weights.push(scope.getVariable(`sc_w0`, [1, outputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                this.weights.push(scope.getVariable(`sc_b0`, [1, 1, outputSize], "float32", tf.initializers.zeros({})))

                for (let i = 1; i <= layerNum; i++) {
                    inputSize = outputSize
                    outputSize = (stateVectorLen / 2) * (i / layerNum) + stateVectorLen * maxCoderSize * (layerNum - i) / layerNum
                    outputSize = Math.ceil(outputSize)
                    this.weights.push(scope.getVariable(`ae_w${i}`, [Math.ceil(sequenceLen / (layerNum)) + 1, inputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                    this.weights.push(scope.getVariable(`ae_b${i}`, [1, 1, outputSize], "float32", tf.initializers.zeros({})))
                    this.weights.push(scope.getVariable(`sc_w${i}`, [1, outputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                    this.weights.push(scope.getVariable(`sc_b${i}`, [1, 1, outputSize], "float32", tf.initializers.zeros({})))
                }

                inputSize = outputSize
                this.weights.push(scope.getVariable(`coding_w`, [Math.ceil(sequenceLen / (layerNum)) + 1, inputSize, outputSize], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                this.weights.push(scope.getVariable(`coding_b`, [1, 1, outputSize], "float32", tf.initializers.zeros({})))

                inputSize += stateVectorLen * maxCoderSize
                for (let i = 1; i <= layerNum; i++) {
                    inputSize += Math.ceil((stateVectorLen / 2) * (i / layerNum) + stateVectorLen * maxCoderSize * (layerNum - i) / layerNum)
                }

                {
                    this.weights.push(scope.getVariable(`value_w1`, [1, sequenceLen, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                    this.weights.push(scope.getVariable(`value_b1`, [1, 1, 1], "float32", tf.initializers.zeros({})))

                    this.weights.push(scope.getVariable(`value_w2`, [1, inputSize, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                    this.weights.push(scope.getVariable(`value_b2`, [1, 1, 1], "float32", tf.initializers.zeros({})))
                }

                {
                    this.weights.push(scope.getVariable(`A_w1`, [1, sequenceLen, 1], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                    this.weights.push(scope.getVariable(`A_b1`, [1, 1, 1], "float32", tf.initializers.zeros({})))

                    this.weights.push(scope.getVariable(`A_w2`, [1, inputSize, actionsNum.reduce((prev, curr) => prev + curr, 0)], "float32", tf.initializers.truncatedNormal({ stddev: 0.1, mean: 0 })))
                    this.weights.push(scope.getVariable(`A_b2`, [1, 1, actionsNum.reduce((prev, curr) => prev + curr, 0)], "float32", tf.initializers.zeros({})))
                }

                for (let i = 1; i <= layerNum; i++) {
                    outputSize = (stateVectorLen / 2) * ((layerNum - i) / layerNum) + stateVectorLen * maxCoderSize * i / layerNum
                    outputSize = Math.ceil(outputSize)
                    this.weights.push(scope.getVariable(`ad_b${i}`, [1, 1, outputSize], "float32", tf.initializers.zeros({})))
                }

                this.weights.push(scope.getVariable(`adOutput_b`, [1, 1, stateVectorLen], "float32", tf.initializers.zeros({})))
                // this.weights.forEach(w => w.sum().print())
                console.log(this.scope.variables)
            }

            predict(x) {
                return tf.tidy(() => {
                    let shotcutLayers = []
                    let AELayer = tf.conv1d(
                        x,
                        this.scope.getVariable(`input_w`),
                        1,
                        "same"
                    )
                    AELayer = tf.add(AELayer, this.scope.getVariable(`input_b`))
                    AELayer = tfex.funcs.mish(AELayer)
                    shotcutLayers.push(
                        tfex.funcs.mish(
                            tf.conv1d(
                                AELayer,
                                this.scope.getVariable(`sc_w0`),
                                1,
                                "same"
                            ).add(this.scope.getVariable(`sc_b0`))
                        )
                    )

                    for (let i = 1; i <= layerNum; i++) {
                        AELayer = tf.conv1d(
                            AELayer,
                            this.scope.getVariable(`ae_w${i}`),
                            1,
                            "same"
                        )
                        AELayer = tf.add(AELayer, this.scope.getVariable(`ae_b${i}`))
                        AELayer = tfex.funcs.mish(AELayer)

                        shotcutLayers.push(
                            tfex.funcs.mish(
                                tf.conv1d(
                                    AELayer,
                                    this.scope.getVariable(`sc_w${i}`),
                                    1,
                                    "same"
                                ).add(this.scope.getVariable(`sc_b${i}`))
                            )
                        )
                    }

                    AELayer = tf.conv1d(
                        AELayer,
                        this.scope.getVariable(`coding_w`),
                        1,
                        "same"
                    )
                    AELayer = tf.add(AELayer, this.scope.getVariable(`coding_b`))
                    AELayer = tfex.funcs.mish(AELayer)

                    let value = tf.concat([AELayer, ...shotcutLayers], 2)
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

                    let A = tf.concat([AELayer, ...shotcutLayers], 2)

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

            predictWithDecoder(x) {
                return tf.tidy(() => {
                    let shotcutLayers = []
                    let AELayer = tf.conv1d(
                        x,
                        this.scope.getVariable(`input_w`),
                        1,
                        "same"
                    )
                    AELayer = tf.add(AELayer, this.scope.getVariable(`input_b`))
                    AELayer = tfex.funcs.mish(AELayer)
                    shotcutLayers.push(
                        tfex.funcs.mish(
                            tf.conv1d(
                                AELayer,
                                this.scope.getVariable(`sc_w0`),
                                1,
                                "same"
                            ).add(this.scope.getVariable(`sc_b0`))
                        )
                    )

                    for (let i = 1; i <= layerNum; i++) {
                        AELayer = tf.conv1d(
                            AELayer,
                            this.scope.getVariable(`ae_w${i}`),
                            1,
                            "same"
                        )
                        AELayer = tf.add(AELayer, this.scope.getVariable(`ae_b${i}`))
                        AELayer = tfex.funcs.mish(AELayer)
                        shotcutLayers.push(
                            tfex.funcs.mish(
                                tf.conv1d(
                                    AELayer,
                                    this.scope.getVariable(`sc_w${i}`),
                                    1,
                                    "same"
                                ).add(this.scope.getVariable(`sc_b${i}`))
                            )
                        )
                    }

                    AELayer = tf.conv1d(
                        AELayer,
                        this.scope.getVariable(`coding_w`),
                        1,
                        "same"
                    )
                    AELayer = tf.add(AELayer, this.scope.getVariable(`coding_b`))
                    AELayer = tfex.funcs.mish(AELayer)

                    let value = tf.concat([AELayer, ...shotcutLayers], 2)
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

                    let A = tf.concat([AELayer, ...shotcutLayers], 2)

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

                    AELayer = tf.add(AELayer, shotcutLayers.pop())
                    for (let i = 0; i < layerNum; i++) {
                        AELayer = tf.conv1d(
                            AELayer,
                            tf.transpose(
                                this.scope.getVariable(`ae_w${layerNum - i}`),
                                [0, 2, 1]
                            ),
                            1,
                            "same"
                        )
                        AELayer = tf.add(AELayer, this.scope.getVariable(`ad_b${i + 1}`))
                        AELayer = tfex.funcs.mish(AELayer)
                        AELayer = tf.add(AELayer, shotcutLayers.pop())
                    }

                    let AELayerOutput = tf.conv1d(
                        AELayer,
                        tf.transpose(
                            this.scope.getVariable(`input_w`),
                            [0, 2, 1]
                        ),
                        1,
                        "same"
                    )
                    AELayerOutput = tf.add(AELayerOutput, this.scope.getVariable(`adOutput_b`))

                    return [outputs, AELayerOutput]
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
            let [evalNet, evalDecoder] = this.model.predictWithDecoder(batchPrevS)
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

            let [predictions, predictionDecoder] = this.model.predictWithDecoder(batchNextS)
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
            return [targetQs, Qs, evalDecoder, predictionDecoder]
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
                            let [targetQs, Qs, evalDecoder, predictionDecoder] = this.tQandQ(
                                batchPrevS,
                                batchAs,
                                batchRs,
                                batchNextS,
                                batchDiscount
                            )
                            let loss = tf.addN(
                                [
                                    tf.mean(
                                        tf.stack(
                                            this.actionsNum.map((actionNum, actionType) => {
                                                // return tf.losses.huberLoss(targetQs[actionType], Qs[actionType])
                                                return tf.losses.meanSquaredError(targetQs[actionType], Qs[actionType])
                                            })
                                        )
                                    ),
                                    tf.losses.huberLoss(
                                        tf.concat([batchPrevS, batchNextS]),
                                        tf.concat([evalDecoder, predictionDecoder])
                                    ),
                                    // ...this.model.getWeights(true).map(w => tf.regularizers.l2().apply(w))
                                ]
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

                    {
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
                                this.memory[replayIdxes_[idx]].p = absTD
                            })
                    }

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
        maxCoderSize
    })
}