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
        discount = 0.63
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
        let stateSeqLayer = input

        for (let i = 0; i < Math.ceil(layerNum / 4); i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen + Math.ceil((stateVectorLen * 3) * (Math.ceil(layerNum / 4) - i) / Math.ceil(layerNum / 4)),
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        for (let i = 0; i < Math.ceil(layerNum / 4); i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: Math.ceil(stateVectorLen / 2 + (stateVectorLen / 2) * (Math.ceil(layerNum / 4) - i) / Math.ceil(layerNum / 4)),
                kernelSize: [Math.ceil(sequenceLen / Math.ceil(layerNum / 4)) + 1],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        for (let i = 1; i <= Math.ceil(layerNum / 4); i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: Math.ceil(stateVectorLen / 2 + (stateVectorLen / 2) * i / Math.ceil(layerNum / 4)),
                kernelSize: [Math.ceil(sequenceLen / Math.ceil(layerNum / 4)) + 1],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        for (let i = 1; i <= Math.ceil(layerNum / 4); i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen + Math.ceil((stateVectorLen * 3) * (i) / Math.ceil(layerNum / 4)),
                kernelSize: [1],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        let value = stateSeqLayer
        {
            value = tf.layers.permute({
                dims: [2, 1],
                inputShape: [sequenceLen, stateVectorLen * 4]
            }).apply(value)
            value = tf.layers.conv1d({
                filters: 1,
                kernelSize: [1]
            }).apply(value)
            value = tf.layers.permute({
                dims: [2, 1],
                inputShape: [stateVectorLen * 4, 1],
            }).apply(value)
            value = tf.layers.conv1d({
                filters: 1,
                kernelSize: [1]
            }).apply(value)
            value = tf.layers.flatten().apply(value)
        }

        let A = stateSeqLayer

        {
            A = tf.layers.permute({
                dims: [2, 1],
                inputShape: [sequenceLen, stateVectorLen * 4]
            }).apply(A)
            A = tf.layers.conv1d({
                filters: 1,
                kernelSize: [1]
            }).apply(A)
            A = tf.layers.permute({
                dims: [2, 1],
                inputShape: [stateVectorLen * 4, 1]
            }).apply(A)
            A = tf.layers.conv1d({
                filters: actionsNum.reduce((prev, curr) => prev + curr, 0),
                kernelSize: [1]
            }).apply(A)
            A = tf.layers.flatten().apply(A)

            A = tfex.layers.lambda({
                func: (x) => {
                    return tf.sub(x, tf.mean(x, 1, true))
                }
            }).apply([A])
        }

        let Q = tf.layers.add().apply([value, A])

        let outputs = tfex.layers.lambda({
            func: (outputLayer) => {
                if (actionsNum.length == 1) {
                    return outputLayer
                } else {
                    return tf.split(outputLayer, actionsNum, 1)
                }
            },
            outputShape: actionsNum.map(actionNum => [null, actionNum])
        }).apply([Q])


        return tf.model({ inputs: [input], outputs: outputs })
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
    discount = 0.63
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
        discount
    })
}