import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"

class Mish extends tf.layers.Layer {
    constructor(config: { inputShape?: tf.Shape }) {
        super(config)
    }
    call(inputs: tf.Tensor | tf.Tensor[]) {
        let input: tf.Tensor
        if (inputs instanceof Array) {
            input = inputs[0]
        } else {
            input = inputs
        }
        return tf.tidy(() => {
            return nn.mish(input)
        })
    }
    static get className() {
        return "Mish"
    }
}
tf.serialization.registerClass(Mish)

export const mish = (config: { inputShape?: tf.Shape }) => new Mish(config)

class Lambda extends tf.layers.Layer {
    constructor(config: { inputShape?: tf.Shape; outputShape?: tf.Shape; fn: (x: tf.Tensor) => tf.Tensor }) {
        super({ inputShape: config.inputShape })
        this.__fn__ = config.fn
        this.__outputShape__ = config.outputShape
    }
    private __fn__: (x: tf.Tensor) => tf.Tensor
    private __outputShape__?: tf.Shape

    call(inputs: tf.Tensor | tf.Tensor[]) {
        let input: tf.Tensor
        if (inputs instanceof Array) {
            input = inputs[0]
        } else {
            input = inputs
        }
        return tf.tidy(() => {
            return this.__fn__(input)
        })
    }

    computeOutputShape(inputShape: tf.Shape) {
        if (this.__outputShape__) {
            return [null, ...this.__outputShape__]
        }
        return [null, ...inputShape]
    }
    static get className() {
        return "Lambda"
    }
}
tf.serialization.registerClass(Lambda)
export const lambda = (config: { inputShape?: tf.Shape; outputShape?: tf.Shape; fn: (x: tf.Tensor) => tf.Tensor }) =>
    new Lambda(config)

class LambdaNetwork extends tf.layers.Layer {
    constructor(config: { dk: number; r: number; heads?: number; dout: number; du?: number }) {
        super({})

        this.__dout__ = config.dout
        this.__du__ = config.du !== undefined ? config.du : 1
        this.__heads__ = config.heads !== undefined ? config.heads : 4

        if (this.__dout__ % this.__heads__ != 0)
            throw new Error("values dimension must be divisible by number of heads for multi-head query")
        this.__dv__ = this.__dout__ / this.__heads__
        this.__dk__ = config.dk

        this.__toQ__ = tf.layers.conv2d({ filters: this.__dk__ * this.__heads__, kernelSize: 1, useBias: false })
        this.__toK__ = tf.layers.conv2d({ filters: this.__dk__ * this.__du__, kernelSize: 1, useBias: false })
        this.__toV__ = tf.layers.conv2d({ filters: this.__dv__ * this.__du__, kernelSize: 1, useBias: false })

        this.__normQ__ = tf.layers.batchNormalization()
        this.__normV__ = tf.layers.batchNormalization()

        if (config.r % 2 == 0) throw new Error("Receptive kernel size should be odd")
        this.__posConv__ = tf.layers.conv3d({
            filters: this.__dk__,
            kernelSize: [1, config.r, config.r],
            padding: "same",
        })
        this._trainableWeights = []
    }
    private __dout__: number
    private __du__: number
    private __heads__: number
    private __dv__: number
    private __dk__: number
    private __toQ__: tf.layers.Layer
    private __toK__: tf.layers.Layer
    private __toV__: tf.layers.Layer
    private __normQ__: tf.layers.Layer
    private __normV__: tf.layers.Layer
    private __posConv__: tf.layers.Layer
    _trainableWeights: tf.LayerVariable[]

    build(inputShape: tf.Shape) {
        let b: number, h: number, w: number, c: number
        if (inputShape.length == 4) [b, h, w, c] = <[number, number, number, number]>inputShape
        else if (inputShape.length == 3) [h, w, c] = <[number, number, number]>inputShape
        else throw new Error("Invalid input shape")

        this.__toQ__.build([null, h, w, c])
        this.__toQ__.weights.map((w) => this._trainableWeights.push(w))
        this.__toK__.build([null, h, w, c])
        this.__toK__.weights.map((w) => this._trainableWeights.push(w))
        this.__toV__.build([null, h, w, c])
        this.__toV__.weights.map((w) => this._trainableWeights.push(w))

        this.__normQ__.build([null, h, w, this.__dk__ * this.__heads__])
        this.__normQ__.weights.map((w) => this._trainableWeights.push(w))
        this.__normV__.build([null, h, w, this.__dv__ * this.__du__])
        this.__normV__.weights.map((w) => this._trainableWeights.push(w))

        this.__posConv__.build([null, this.__dv__, h, w, this.__du__])
        this.__posConv__.weights.map((w) => this._trainableWeights.push(w))

        this.built = true
    }

    call(inputs: tf.Tensor | tf.Tensor[]) {
        let input: tf.Tensor
        if (inputs instanceof Array) {
            input = inputs[0]
        } else {
            input = inputs
        }
        return tf.tidy(() => {
            const [b, h, w, c, du, dk, dv, heads, dout] = [
                ...(<tf.Tensor4D>input).shape,
                this.__du__,
                this.__dk__,
                this.__dv__,
                this.__heads__,
                this.__dout__,
            ]
            const n = h * w

            let q = <tf.Tensor4D>this.__toQ__.apply(input)
            let k = <tf.Tensor4D>this.__toK__.apply(input)
            let v = <tf.Tensor4D>this.__toV__.apply(input)

            q = <tf.Tensor4D>this.__normQ__.apply(q)
            v = <tf.Tensor4D>this.__normV__.apply(v)

            q = q.reshape([b, n, heads, dk]).transpose([0, 2, 3, 1])
            k = k.reshape([b, n, du, dk]).transpose([0, 2, 3, 1])
            v = v.reshape([b, n, du, dv]).transpose([0, 2, 3, 1])

            k = tf.softmax(k, -1)

            let Lc = <tf.Tensor3D>tf.mul(k.reshape([b, du, dk, 1, n]), v.reshape([b, du, 1, dv, n])).sum([1, 4])
            let Yc = <tf.Tensor4D>tf
                .mul(q.reshape([b, heads, dk, n, 1]), Lc.reshape([b, 1, dk, 1, dv]))
                .sum(2)
                .transpose([0, 2, 1, 3])

            v = v.transpose([0, 2, 3, 1]).reshape([b, dv, h, w, du])
            let Lp = <tf.Tensor5D>this.__posConv__.apply(v)
            Lp = Lp.reshape([b, dv, n, dk]).transpose([0, 1, 3, 2])
            let Yp = <tf.Tensor4D>tf
                .mul(q.reshape([b, heads, 1, dk, n]), Lp.reshape([b, 1, dv, dk, n]))
                .sum(3)
                .transpose([0, 3, 1, 2])

            let Y = Yc.add(Yp)
            let out = Y.reshape([b, h, w, dout])

            return out
        })
    }

    computeOutputShape(inputShape: tf.Shape) {
        if (inputShape.length == 4) return [...inputShape.slice(0, 3), this.__dout__]
        else if (inputShape.length == 3) return [null, ...inputShape.slice(0, 2), this.__dout__]
        else throw new Error("Invalid input shape")
    }
    static get className() {
        return "LambdaNetwork"
    }
}

tf.serialization.registerClass(LambdaNetwork)
export const lambdaNetwork = (config: { dk: number; r: number; heads?: number; dout: number; du?: number }) =>
    new LambdaNetwork(config)
