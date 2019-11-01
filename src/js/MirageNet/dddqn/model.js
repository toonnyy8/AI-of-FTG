import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../../lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

export class DDDQN {
    constructor({
        sequenceLen = 60,
        inputNum = 10,
        embInner = [32, 32, 32],
        filters = [8, 8, 8, 8],
        outputInner = [32, 32],
        actionNum = 8,
        memorySize = 1000,
        updateTargetStep = 20,
        learningRate = 1e-3
    }) {

        {
            this.updateTargetStep = updateTargetStep

            this.count = 0

            this.actionNum = actionNum
        }

        {
            this.model = this.buildModel2({
                sequenceLen: sequenceLen,
                inputNum: inputNum,
                embInner: embInner,
                filters: filters,
                outputInner: outputInner,
                actionNum: actionNum
            })
            this.model.summary()

            this.targetModel = this.buildModel2({
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
            this.optimizer = tf.train.adam(learningRate)
        }

    }

    buildModel(
        {
            sequenceLen,
            inputNum,
            embInner = [64, 64, 64],
            filters = [64, 64, 64],
            outputInner = [64, 64],
            actionNum = 36
        }
    ) {
        let input = tf.input({ shape: [sequenceLen, inputNum] })
        let preASV = tf.input({ shape: [actionNum] })

        let embLayer = tf.layers.dense({ units: embInner[0], activation: 'selu' }).apply(input)
        for (let i = 1; i < embInner.length; i++) {
            embLayer = tf.layers.dense({ units: embInner[i], activation: 'selu' }).apply(embLayer)
        }
        embLayer = tf.layers.reshape({ targetShape: [sequenceLen, embInner[embInner.length - 1], 1] }).apply(embLayer)
        embLayer = tf.layers.dropout({ rate: 0.1 }).apply(embLayer)

        let cnnLayer = tf.layers.conv2d({
            filters: filters[0],
            kernelSize: [2, embInner[embInner.length - 1]],
            activation: "selu",
            padding: "same"
        }).apply(embLayer)
        for (let i = 1; i < filters.length - 1; i++) {
            cnnLayer = tf.layers.conv2d({
                filters: filters[i],
                kernelSize: [i * Math.floor(sequenceLen / filters.length), embInner[embInner.length - 1]],
                activation: "selu",
                padding: "same"
            }).apply(cnnLayer)
        }
        cnnLayer = tf.layers.conv2d({
            filters: filters[filters.length - 1],
            kernelSize: [sequenceLen, embInner[embInner.length - 1]],
            strides: [sequenceLen, embInner[embInner.length - 1]],
            activation: "selu",
            padding: "same"
        }).apply(cnnLayer)

        let flattenLayer = tf.layers.flatten().apply(cnnLayer)
        flattenLayer = tf.layers.dropout({ rate: 0.1 }).apply(flattenLayer)

        let outputLayer = tf.layers.dense({ units: outputInner[0], activation: 'selu' }).apply(flattenLayer)
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

        let mean = tfex.layers.lambda({
            func: (x) => {
                return tf.mean(x, 1, true)
            },
            outputShape: [1]
        }).apply([A])

        let advantage = tfex.layers.lambda({
            func: (x, y) => {
                return tf.sub(x, y)
            }
        }).apply([A, mean])

        let Q = tf.layers.add().apply([value, advantage])

        //Action Selection Value
        let ASV = tf.layers.softmax().apply(Q)

        //Action Activation Value
        let AAV = tfex.layers.lambda({
            func: (ASV, preASV) => {
                return tf.div(tf.sub(ASV, preASV), tf.max(tf.stack([ASV, preASV]), 0))
            }
        }).apply([ASV, preASV])

        // AAV = tf.layers.softmax().apply(AAV)

        class WeightedAverage extends tf.layers.Layer {
            constructor(args) {
                super({})
            }
            build(inputShape) {
                // console.log("LayerNorm build : ")
                this.w = this.addWeight("w", [inputShape[0][inputShape.length - 1]], "float32", tf.initializers.constant({ value: 0.5 }))
                this.built = true
            }
            computeOutputShape(inputShape) {
                //console.log("LayerNorm computeOutputShape")
                //console.log(inputShape)
                return inputShape[0]
            }
            call(inputs, kwargs) {
                //console.log("LayerNorm call")
                this.invokeCallHook(inputs, kwargs)
                return tf.add(
                    tf.mul(inputs[0], this.w.read()),
                    tf.mul(inputs[1], tf.sub(1, this.w.read()))
                )
            }

            /*
            * If a custom layer class is to support serialization, it must implement
            * the `className` static getter.
            */
            static get className() {
                return "WeightedAverage"
            }
        }
        // registerClass
        tf.serialization.registerClass(WeightedAverage)

        let action = new WeightedAverage().apply([ASV, AAV])
        // action = tf.layers.softmax().apply(action)

        return tf.model({ inputs: [input, preASV], outputs: [ASV, action] })
    }

    buildModel2(
        {
            sequenceLen,
            inputNum,
            filters = 8,
            actionNum = 36
        }
    ) {
        let input = tf.input({ shape: [sequenceLen, inputNum] })
        let preASV = tf.input({ shape: [actionNum] })

        class WeightedSequence extends tf.layers.Layer {
            constructor(args = { axis, script }) {
                super({})
                this.axis = args.axis
                this.script = args.script
            }
            build(inputShape) {
                this.w = this.addWeight("w", [inputShape[this.axis]], "float32", tf.initializers.constant({ value: 0.5 }))
                this.w.write(tf.sin(tf.linspace(Math.PI / 2, 0.1, inputShape[this.axis])))
                this.built = true
            }
            computeOutputShape(inputShape) {
                return inputShape
            }
            call(inputs, kwargs) {
                //console.log("LayerNorm call")
                this.invokeCallHook(inputs, kwargs)
                return tfex.funcs.einsum(this.script, inputs[0], this.w.read())
            }

            /*
            * If a custom layer class is to support serialization, it must implement
            * the `className` static getter.
            */
            static get className() {
                return "WeightedSequence"
            }
        }
        // registerClass
        tf.serialization.registerClass(WeightedSequence)

        let WSLayer = new WeightedSequence({ axis: 1, script: "ijk,j->ijk" }).apply(input)

        let cnnLayer = tf.layers.conv1d({
            filters: filters * 4,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(WSLayer)
        cnnLayer = tf.layers.conv1d({
            filters: filters * 4,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(cnnLayer)

        cnnLayer = tf.layers.dropout({ rate: 0.1 }).apply(cnnLayer)

        while (1 <= cnnLayer.shape[1] / 2) {
            cnnLayer = tf.layers.conv1d({
                filters: filters,
                kernelSize: [4],
                strides: [2],
                activation: "selu",
                padding: "same"
            }).apply(cnnLayer)
            cnnLayer = tf.layers.batchNormalization({}).apply(cnnLayer)
        }

        let value = tf.layers.conv1d({
            filters: filters * 4,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(cnnLayer)
        value = tf.layers.conv1d({
            filters: filters * 4,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(value)

        value = tf.layers.conv1d({
            filters: actionNum,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(value)

        let A = tf.layers.conv1d({
            filters: filters * 4,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(cnnLayer)
        A = tf.layers.conv1d({
            filters: filters * 4,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(A)

        A = tf.layers.conv1d({
            filters: actionNum,
            kernelSize: [1],
            activation: "selu",
            padding: "same"
        }).apply(A)

        let mean = tfex.layers.lambda({
            func: (x) => {
                return tf.mean(x, 1, true)
            },
            outputShape: [1]
        }).apply([A])

        let advantage = tfex.layers.lambda({
            func: (x, y) => {
                return tf.sub(x, y)
            }
        }).apply([A, mean])

        let Q = tf.layers.flatten().apply(
            tf.layers.add().apply([value, advantage])
        )

        //Action Selection Value
        let ASV = tf.layers.softmax().apply(Q)

        //Action Activation Value
        let AAV = tfex.layers.lambda({
            func: (ASV, preASV) => {
                return tf.tidy(() => {
                    let aav = tf.sub(ASV, preASV)
                    aav = tf.relu(aav)
                    aav = tf.div(aav, aav.sum(1, true))
                    return aav
                })
            }
        }).apply([ASV, preASV])

        // AAV = tf.layers.softmax().apply(AAV)

        class WeightedAverage extends tf.layers.Layer {
            constructor(args) {
                super({})
            }
            build(inputShape) {
                // console.log("LayerNorm build : ")
                this.w = this.addWeight("w", [inputShape[0][inputShape.length - 1]], "float32", tf.initializers.constant({ value: 0.8 }))
                this.built = true
            }
            computeOutputShape(inputShape) {
                //console.log("LayerNorm computeOutputShape")
                //console.log(inputShape)
                return inputShape[0]
            }
            call(inputs, kwargs) {
                //console.log("LayerNorm call")
                this.invokeCallHook(inputs, kwargs)
                return tf.add(
                    tf.mul(inputs[0], this.w.read()),
                    tf.mul(inputs[1], tf.sub(1, this.w.read()))
                )
            }

            /*
            * If a custom layer class is to support serialization, it must implement
            * the `className` static getter.
            */
            static get className() {
                return "WeightedAverage"
            }
        }
        // registerClass
        tf.serialization.registerClass(WeightedAverage)

        let action = new WeightedAverage().apply([ASV, AAV])

        return tf.model({ inputs: [input, preASV], outputs: [ASV, action] })
    }

    loss(arrayPrevS, arrayPrevASV, arrayA, arrayR, arrayNextS, arrayNextASV) {
        return tf.tidy(() => {
            // console.log(arrayPrevS)
            let batchPrevS = tf.tensor3d(arrayPrevS)
            let batchPrevASV = tf.tensor2d(arrayPrevASV)
            let batchA = tf.tensor1d(arrayA, 'int32')
            let batchR = tf.tensor1d(arrayR)
            let batchNextS = tf.tensor3d(arrayNextS)
            let batchNextASV = tf.tensor2d(arrayNextASV)

            const predictions = this.model.predict([batchPrevS, batchPrevASV]);

            const maxQ = this.targetModel.predict([batchNextS, predictions[0]])[1].reshape([arrayPrevS.length, this.actionNum]).max(1)

            const predMask = tf.oneHot(batchA, this.actionNum);

            const targets = batchR.add(maxQ.mul(tf.scalar(0.99)));

            return tf.mul(predictions[1].sub(targets.expandDims(1)).square(), predMask.asType('float32')).mean();
        })

    }

    train(replayNum = 100, loadIdxes = [null], usePrioritizedReplay = false) {
        tf.tidy(() => {
            let train_ = (replayIdxes) => {
                tf.tidy(() => {
                    let arrayPrevS = []
                    let arrayPrevASV = []
                    let arrayA = []
                    let arrayR = []
                    let arrayNextS = []
                    let arrayNextASV = []

                    for (let i = 0; i < replayNum; i++) {
                        let data = this.load(replayIdxes[i])
                        // console.log(data)
                        arrayPrevS.push(data[0])
                        arrayPrevASV.push(data[1])
                        arrayA.push(data[2])
                        arrayR.push(data[3])
                        arrayNextS.push(data[4])
                        arrayNextASV.push(data[5])
                    }

                    let grads = this.optimizer.computeGradients(
                        () => {
                            let loss = this.loss(arrayPrevS, arrayPrevASV, arrayA, arrayR, arrayNextS, arrayNextASV)
                            loss.print()
                            return loss
                        }, this.model.getWeights(true)).grads

                    let gradsName = Object.keys(grads)
                    grads = tfex.funcs.clipByGlobalNorm(Object.values(grads), 0.05)[0]

                    this.optimizer.applyGradients(gradsName.reduce((acc, gn, idx) => {
                        acc[gn] = grads[idx]
                        return acc
                    }, {}))

                    this.count++

                    if (this.count >= this.updateTargetStep) {

                        this.targetModel.setWeights(this.model.getWeights())
                        this.count = 0
                    }
                })
            }
            if (this.memory.length != 0) {
                if (usePrioritizedReplay) {
                    tf.tidy(() => {
                        let e = tf.tensor(this.memory.map(mem => mem[3]))
                        e = tf.abs(e.sub(e.mean()))
                        e = e.div(e.sum(0, true))
                        // e.print()
                        return tf.multinomial(e, replayNum, null, true)
                    }).array().then(prioritizedReplayBuffer => {
                        // console.log(prioritizedReplayBuffer)
                        train_(prioritizedReplayBuffer.map((prioritizedReplayIdx, idx) => {
                            return loadIdxes[idx] == null || loadIdxes[idx] == undefined ? prioritizedReplayIdx : loadIdxes[idx]
                        }))
                    })
                } else {
                    train_(loadIdxes)
                }
            }
        })
    }

    store(preState, preASV, action, reward, nextState, nextASV) {
        if (this.memory.length == this.memorySize) {
            this.memory.pop()
        }
        this.memory.unshift([preState, preASV, action, reward, nextState, nextASV])
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
    embInner = [32, 32, 32],
    filters = [8, 8, 8, 8],
    outputInner = [32, 32],
    actionNum = 8,
    memorySize = 1000,
    updateTargetStep = 20,
    learningRate = 1e-3
}) {
    return new DDDQN({
        sequenceLen,
        inputNum,
        embInner,
        filters,
        outputInner,
        actionNum,
        memorySize,
        updateTargetStep,
        learningRate
    })
}