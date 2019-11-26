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
        initLearningRate = 1e-3,
        minLearningRate = 1e-5,
        discount = 0.99
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
        let cbam1d = (inputLayer) => {
            let x_mean = tf.layers.globalAveragePooling1d({}).apply(inputLayer) // (B, C)
            x_mean = tf.layers.reshape({ targetShape: [1, x_mean.shape[1]] }).apply(x_mean) // (B, 1, C)
            x_mean = tf.layers.conv1d({
                    filters: Math.ceil(stateVectorLen / 2),
                    kernelSize: 1,
                    strides: 1,
                    activation: "selu",
                }).apply(x_mean) //(B, 1, C // r)
            x_mean = tf.layers.conv1d({
                    filters: stateVectorLen,
                    kernelSize: 1,
                    strides: 1,
                    activation: "selu",
                }).apply(x_mean) //(B, 1, C)

            let x_max = tf.layers.globalMaxPooling1d({}).apply(inputLayer) // (B, C)
            x_max = tf.layers.reshape({ targetShape: [1, x_max.shape[1]] }).apply(x_max) // (B, 1, C)
            x_max = tf.layers.conv1d({
                    filters: Math.ceil(stateVectorLen / 2),
                    kernelSize: 1,
                    strides: 1,
                    activation: "selu",
                }).apply(x_max) //(B, 1, C // r)
            x_max = tf.layers.conv1d({
                    filters: stateVectorLen,
                    kernelSize: 1,
                    strides: 1,
                    activation: "selu",
                }).apply(x_max) //(B, 1, C)

            let x = tf.layers.add({}).apply([x_mean, x_max]) // (B, 1, C)
            x = tfex.layers.lambda({ func: (x) => { return tf.sigmoid(x) }, outputShape: x.shape }).apply(x) // (B, 1, C)
            x = tf.layers.multiply().apply([inputLayer, x]) // (B, T, C)

            let x_ = tf.layers.permute({
                    dims: [2, 1],
                    inputShape: [x.shape[1], x.shape[2]]
                }).apply(x) // (B, C, T)
                // spatial attention
            let y_mean = tf.layers.globalAveragePooling1d({}).apply(x_) // (B, T)
            y_mean = tf.layers.reshape({ targetShape: [x.shape[1], 1] }).apply(y_mean) // (B, T, 1)

            let y_max = tf.layers.globalMaxPooling1d({}).apply(x_) // (B, W*H)
            y_max = tf.layers.reshape({ targetShape: [x.shape[1], 1] }).apply(y_max) // (B, T, 1)

            let y = tfex.layers.lambda({
                    func: (y_mean, y_max) => { return tf.concat([y_mean, y_max], 2) },
                    outputShape: [null, x.shape[1], 2]
                }).apply([y_mean, y_max]) // (B, T, 2)

            y = tf.layers.conv1d({
                filters: 1,
                kernelSize: 7,
                activation: "sigmoid",
                padding: "same"
            }).apply(y)
            y = tf.layers.multiply().apply([x, y]) // (B, W, H, C)
            return y
        }

        let input = tf.input({ shape: [sequenceLen, stateVectorLen] })

        let stateSeqLayer = tf.layers.conv1d({
            filters: stateVectorLen,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(input)

        for (let i = 0; i < Math.ceil(layerNum / 4); i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen,
                kernelSize: [4],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        stateSeqLayer = tf.layers.add({}).apply([cbam1d(stateSeqLayer), stateSeqLayer])

        stateSeqLayer = tf.layers.conv1d({
            filters: stateVectorLen,
            kernelSize: [2],
            strides: [2]
        }).apply(stateSeqLayer)

        for (let i = 0; i < Math.ceil(layerNum / 4); i++) {
            stateSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen,
                kernelSize: [4],
                activation: "selu",
                padding: "same"
            }).apply(stateSeqLayer)
        }

        stateSeqLayer = tf.layers.add({}).apply([cbam1d(stateSeqLayer), stateSeqLayer])

        stateSeqLayer = tf.layers.conv1d({
            filters: stateVectorLen,
            kernelSize: [2],
            strides: [2]
        }).apply(stateSeqLayer)

        let outputs = actionsNum.map(actionNum => {
            let actionSeqLayer = stateSeqLayer

            for (let i = 0; i < Math.ceil(layerNum / 4); i++) {
                actionSeqLayer = tf.layers.conv1d({
                    filters: stateVectorLen,
                    kernelSize: [4],
                    activation: "selu",
                    padding: "same"
                }).apply(actionSeqLayer)
            }

            actionSeqLayer = tf.layers.add({}).apply([cbam1d(actionSeqLayer), actionSeqLayer])

            actionSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen,
                kernelSize: [2],
                strides: [2]
            }).apply(actionSeqLayer)

            for (let i = 0; i < Math.ceil(layerNum / 4); i++) {
                actionSeqLayer = tf.layers.conv1d({
                    filters: stateVectorLen,
                    kernelSize: [4],
                    activation: "selu",
                    padding: "same"
                }).apply(actionSeqLayer)
            }

            actionSeqLayer = tf.layers.add({}).apply([cbam1d(actionSeqLayer), actionSeqLayer])

            actionSeqLayer = tf.layers.conv1d({
                filters: stateVectorLen,
                kernelSize: actionSeqLayer.shape[1],
                strides: actionSeqLayer.shape[1],
                activation: "selu",
            }).apply(actionSeqLayer)

            let value = actionSeqLayer

            {
                value = tf.layers.conv1d({
                    filters: 1,
                    kernelSize: [1],
                    activation: "selu",
                    padding: "same"
                }).apply(value)
                value = tf.layers.flatten().apply(value)
            }

            let A = actionSeqLayer

            {
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
                const targets = batchRs[actionType].add(maxQ.mul(tf.scalar(this.discount)));
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
                        return tf.tensor1d(arrayR, "float32")
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
    sequenceLen = 60,
    stateVectorLen = 10,
    embInner = [32, 32, 32],
    layerNum = 8,
    outputInner = [32, 32],
    actionsNum = [3, 3, 4],
    memorySize = 1000,
    updateTargetStep = 0.05,
    initLearningRate = 1e-3,
    minLearningRate = 1e-5,
    discount = 0.99
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
        initLearningRate,
        minLearningRate,
        discount
    })
}