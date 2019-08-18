import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"

//----------positional embedding
export function positionalEmbedding(
    args = {
        bsz: null,
        outputShape
    }
) {
    return tfex.layers.lambda({
        func: (posSeq, invFreq) => {
            let sinusoidInp = tfex.einsum('i,j->ij', posSeq, invFreq)
            let posEmb = tf.concat([tf.sin(sinusoidInp), tf.cos(sinusoidInp)], -1)
            if (args.bsz != null) {
                return posEmb.expandDims(1).tile([1, args.bsz, 1])
            }
            else {
                return posEmb.expandDims(1)
            }
        },
        outputShape: args.outputShape
    })
}
//----------positionwise feed forward
export class PositionwiseFF extends tf.layers.Layer {
    constructor(
        args = {
            dModel,
            dInner,
            dropout,
            kernelInitializer,
            isTraining: true
        }
    ) {
        super({})
        console.log(this.args)
        this.dModel = args.dModel
        this.dInner = args.dInner
        this.dropout = args.dropout
        this.kernelInitializer = args.kernelInitializer
        this.isTraining = args.isTraining
        //console.log("PositionwiseFF constructor")

    }

    build(inputShape) {
        //console.log("PositionwiseFF build : ")

        this.w_ = {
            kernel1: this.addWeight(
                "kernel1",
                [
                    inputShape.reduce((last, val) => last * val, 1),
                    this.dInner
                ],
                "float32"
            ),
            bias1: this.addWeight(
                "bias1",
                [
                    this.dInner
                ],
                "float32"
            ),
            kernel2: this.addWeight(
                "kernel2",
                [
                    this.dInner,
                    this.dModel
                ],
                "float32"
            ),
            bias2: this.addWeight(
                "bias2",
                [
                    this.dModel
                ],
                "float32"
            )
        }
        this.built = true
    }

    computeOutputShape(inputShape) {
        //console.log("PositionwiseFF computeOutputShape")
        //console.log(inputShape)
        return inputShape
    }

    call(inputs, kwargs) {
        let input = inputs
        if (Array.isArray(input)) {
            input = input[0]
        }
        //console.log("PositionwiseFF call")
        this.invokeCallHook(inputs, kwargs)

        return tf.tidy(() => {
            return tf.dropout(
                tf.dropout(
                    input
                        .reshape([-1])
                        .dot(this.w_.kernel1)
                        .add(this.w_.bias1),
                    this.dropout
                )
                    .dot(this.w_.kernel2)
                    .add(this.w_.bias2),
                this.dropout
            ).reshape([this.dModel])
        })
    }

    /*
    * If a custom layer class is to support serialization, it must implement
    * the `className` static getter.
    */
    static get className() {
        return "PositionwiseFF"
    }

    getClassName() {
        return "PositionwiseFF"
    }
}
tf.serialization.registerClass(PositionwiseFF)

export function positionwiseFF(
    args = {
        dModel,
        dInner,
        dropout,
        kernelInitializer,
        isTraining: true
    }
) {
    return new PositionwiseFF(args)
}
//----------relative shift
export function relShift(
    args = {
        outputShape
    }
) {
    return tfex.layers.lambda({
        func: (x) => {
            let x_size = x.shape
            x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
            x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
            x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
            x = tf.reshape(x, x_size)
            return x
        },
        outputShape: outputShape
    })
}
//----------relative multihead attnention
export class RelMultiheadAttn extends tf.layers.Layer {
    constructor(
        args = {
            dModel: null,
            nHead: null,
            dHead: null,
            dropout: null,
            dropatt: null,
            isTraining: null,
            kernelInitializer: null
        }
    ) {
        super({})
        console.log(this.args)
        this.dModel = args.dModel
        this.nHead = args.nHead
        this.dHead = args.dHead
        this.dropout = args.dropout
        this.dropatt = args.dropatt
        this.isTraining = args.isTraining
        this.kernelInitializer = args.kernelInitializer
        this.scale = 1 / (args.dHead ** 0.5)

        this.isInput = {
            w: false,
            r: false,
            r_wBias: false,
            r_rBias: false,
            attnMask: false,
            mems: false
        }
        //console.log("RelMultiheadAttn constructor")

    }

    build(inputShape) {
        //console.log("RelMultiheadAttn build : ")
        // [w, r, r_wBias, r_rBias, attnMask, mems]
        console.log(inputShape)
        let catShape = this.isInput.mems && inputShape[5].length > 1 ? inputShape[0].concat(inputShape[5]) : inputShape[0].concat([])
        this.qlen = inputShape[0][0]
        this.rlen = inputShape[1][0]
        this.bsz = inputShape[0][1]
        this.klen = this.nHead * this.dHead * 3
        console.log(this.qlen * this.bsz * this.nHead * this.dHead)
        this.w_ = {
            qkvKernel: this.addWeight(
                "qkvKernel",
                [
                    catShape.reduce((last, val) => last * val, 1),
                    this.nHead * this.dHead * 3
                ],
                "float32",
                this.kernelInitializer
            ),
            qkvBias: this.addWeight(
                "qkvBias",
                [
                    this.nHead * this.dHead * 3
                ],
                "float32",
                this.kernelInitializer
            ),
            rKernel: this.addWeight(
                "rKernel",
                [
                    inputShape[1].reduce((last, val) => last * val, 1),
                    this.nHead * this.dHead
                ],
                "float32",
                this.kernelInitializer
            ),
            rBias: this.addWeight(
                "rBias",
                [
                    this.nHead * this.dHead
                ],
                "float32",
                this.kernelInitializer
            ),
            oKernel: this.addWeight(
                "oKernel",
                [
                    this.qlen * this.bsz * this.nHead * this.dHead,
                    this.dModel
                ],
                "float32",
                this.kernelInitializer
            ),
            oBias: this.addWeight(
                "oBias",
                [
                    this.dModel
                ],
                "float32",
                this.kernelInitializer
            )
        }
        this.built = true
    }

    computeOutputShape(inputShape) {
        // console.log("RelMultiheadAttn computeOutputShape")
        console.log(inputShape)
        return inputShape
    }

    apply(
        args = {
            w: null,
            r: null,
            r_wBias: null,
            r_rBias: null,
            attnMask: null,
            mems: null
        },
        kwargs
    ) {
        return tf.tidy(() => {
            Object.keys(this.isInput).forEach((inputKey) => {
                if (args[inputKey] instanceof tf.Tensor) {
                    this.isInput[inputKey] = true
                } else {
                    args[inputKey] = tf.tensor([false])
                    this.isInput[inputKey] = false
                    console.log(`RelMultiheadAttn : ${inputKey} is not entered or the type is not tensor`)
                }
            })
            let inputs = [args.w, args.r, args.r_wBias, args.r_rBias, args.attnMask, args.mems]
            return super.apply(inputs, kwargs)
        })
    }

    call(inputs, kwargs) {
        let args = {
            w: inputs[0],
            r: inputs[1],
            r_wBias: inputs[2],
            r_rBias: inputs[3],
            attnMask: inputs[4],
            mems: inputs[5]
        }
        console.log("RelMultiheadAttn call")
        this.invokeCallHook(inputs, kwargs)

        return tf.tidy(() => {
            return
        })
    }

    /*
    * If a custom layer class is to support serialization, it must implement
    * the `className` static getter.
    */
    static get className() {
        return "RelMultiheadAttn"
    }

    getClassName() {
        return "RelMultiheadAttn"
    }
}
tf.serialization.registerClass(RelMultiheadAttn)

export function relMultiheadAttn(
    args = {
        dModel,
        nHead,
        dHead,
        dropout,
        dropatt,
        isTraining,
        kernelInitializer
    }
) {
    return new RelMultiheadAttn(args)
}