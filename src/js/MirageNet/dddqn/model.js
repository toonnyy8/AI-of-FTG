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
            this.model.summary()

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
            this.optimizer = tf.train.adam(1e-4)
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

    loss(arrayPrevS, arrayPrevASV, arrayA, arrayR, arrayNextS, arrayNextASV) {
        return tf.tidy(() => {
            console.log(arrayPrevS)
            let batchPrevS = tf.tensor3d(arrayPrevS)
            let batchPrevASV = tf.tensor2d(arrayPrevASV)
            let batchA = tf.tensor1d(arrayA, 'int32')
            let batchR = tf.tensor1d(arrayR)
            let batchNextS = tf.tensor3d(arrayNextS)
            let batchNextASV = tf.tensor2d(arrayNextASV)

            const maxQ = this.targetModel.predict([batchNextS, batchNextASV])[1].reshape([arrayPrevS.length, this.actionNum]).max(1)

            const predMask = tf.oneHot(batchA, this.actionNum);

            const targets = batchR.add(maxQ.mul(tf.scalar(0.99)));

            const predictions = this.model.predict([batchPrevS, batchPrevASV])[1];

            return tf.mul(predictions.sub(targets.expandDims(1)).square(), predMask.asType('float32')).mean();
        })

    }

    train(replayNum = 100) {
        tf.tidy(() => {
            let arrayPrevS = []
            let arrayPrevASV = []
            let arrayA = []
            let arrayR = []
            let arrayNextS = []
            let arrayNextASV = []

            for (let i = 0; i < replayNum; i++) {
                let data = this.load()
                console.log(data)
                arrayPrevS.push(data[0])
                arrayPrevASV.push(data[1])
                arrayA.push(data[2])
                arrayR.push(data[3])
                arrayNextS.push(data[4])
                arrayNextASV.push(data[5])
            }

            this.optimizer.minimize(() => {
                let loss = this.loss(arrayPrevS, arrayPrevASV, arrayA, arrayR, arrayNextS, arrayNextASV)
                // loss.print()
                return loss
            }, true, this.model.getWeights());

            this.count++

            if (this.count >= this.updateTargetStep) {

                this.targetModel.setWeights(this.model.getWeights())
                this.count = 0
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