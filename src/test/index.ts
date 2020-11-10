import * as tf from "@tensorflow/tfjs"
import { MHA, positionalEncoding } from "../cnn_dddqn/model/mha"
import { AED } from "../cnn_dddqn/model/ae"
import * as nn from "./nn"
import { reshape } from "@tensorflow/tfjs";
tf.enableProdMode()
class LambdaLayer extends tf.layers.Layer {
    constructor({
        dim,
        dim_k,
        n = undefined,
        r = undefined,
        heads = 4,
        dim_out = undefined,
        dim_u = 1
    }: {
        dim: number,
        dim_k: number,
        n?: number,
        r?: number,
        heads: number,
        dim_out?: number,
        dim_u: number
    }) {
        super({});
        this.config.out_dim = dim_out
        this.config.u = dim_u
        this.config.heads = heads

        console.assert(((dim_out ? dim_out : 0) % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query')
        this.config.dim_v = Math.round((dim_out ? dim_out : 0) / heads)
        this.config.dim_k = dim_k
        this.config.heads = heads

        this.config.to_q = tf.layers.separableConv2d({ filters: this.config.dim_k * heads, kernelSize: 1, useBias: false, inputShape: [1, 1, dim] })
        this.config.to_k = tf.layers.separableConv2d({ filters: this.config.dim_k * dim_u, kernelSize: 1, useBias: false, inputShape: [1, 1, dim] })
        this.config.to_v = tf.layers.separableConv2d({ filters: this.config.dim_v * dim_u, kernelSize: 1, useBias: false, inputShape: [1, 1, dim] })

        this.config.norm_q = tf.layers.batchNormalization()
        this.config.norm_v = tf.layers.batchNormalization()

        this.config.local_contexts = (r !== undefined) && (r !== 0)
        if (this.config.local_contexts) {
            console.assert((<number>r % 2) == 1, 'Receptive kernel size should be odd')
            this.config.pos_conv = tf.layers.conv3d({
                filters: this.config.dim_k,
                kernelSize: [1, <number>r, <number>r],
                padding: 'same',
            })
            this.config.pos_conv_ = tf.layers.conv3d({
                filters: this.config.dim_k,
                kernelSize: [1, <number>r, <number>r],
                padding: 'same',
            })
        } else {
            console.assert((n !== undefined) && (n !== 0), 'You must specify the total sequence length (h x w)')
            this.config.pos_emb = tf.variable(
                tf.randomNormal([<number>n, <number>n, dim_k, dim_u]),
                true,
                'pos_emb',
                "float32",
            )
        }
    }

    computeOutputShape(input_shape: tf.Shape | tf.Shape[]) {
        return <tf.Shape | tf.Shape[]>[...input_shape.slice(0, 2), this.config.out_dim]
    }

    config: {
        out_dim?: number,
        u?: number,
        heads?: number,
        dim_v?: number,
        dim_k?: number,
        to_q?: tf.layers.Layer,
        to_k?: tf.layers.Layer,
        to_v?: tf.layers.Layer,
        norm_q?: tf.layers.Layer,
        norm_v?: tf.layers.Layer,
        local_contexts?: boolean,
        pos_conv?: tf.layers.Layer,
        pos_conv_?: tf.layers.Layer,
        pos_emb?: tf.Variable,
    } = {}


    /**
     * call() contains the actual numerical computation of the layer.
     *
     * It is "tensor-in-tensor-out". I.e., it receives one or more
     * tensors as the input and should produce one or more tensors as
     * the return value.
     *
     * Be sure to use tidy() to avoid WebGL memory leak. 
     */
    call(x: tf.Tensor) {
        return tf.tidy(() => {
            const [b, height, width, c, u, heads, dim_k, dim_v]
                = <[number, number, number, number, number, number, number, number,]>
                [...x.shape, this.config.u, this.config.heads, this.config.dim_k, this.config.dim_v,]

            let q = nn.pipe(
                nn.layerFn(<tf.layers.Layer>this.config.to_q),
                nn.layerFn(<tf.layers.Layer>this.config.norm_q),
            )(x)
            q = q.reshape([b, height * width, heads, dim_k]).transpose([0, 2, 3, 1])

            let k = nn.layerFn(<tf.layers.Layer>this.config.to_k)(x)
            k = k.reshape([b, height * width, u, dim_k]).transpose([0, 2, 3, 1])
            k = tf.softmax(k, -1)

            let v = nn.pipe(
                nn.layerFn(<tf.layers.Layer>this.config.to_v),
                nn.layerFn(<tf.layers.Layer>this.config.norm_v),
            )(x)
            v = v.reshape([b, height * width, u, dim_v]).transpose([0, 2, 3, 1])

            const Lc = tf.mul(
                k.reshape([b, u, dim_k, 1, height * width]),
                v.reshape([b, u, 1, dim_v, height * width])
            ).sum([1, 4])

            const Yc = tf.mul(
                q.reshape([b, heads, dim_k, 1, height * width]),
                Lc.reshape([b, 1, dim_k, dim_v, 1])
            ).sum([2])
                .transpose([0, 3, 1, 2])

            const Yp = (() => {
                if (this.config.local_contexts) {
                    v = v.reshape([b, u, dim_v, height, width]).transpose([0, 2, 3, 4, 1])
                    let Lp = nn.pipe(
                        nn.layerFn(<tf.layers.Layer>this.config.pos_conv),
                        nn.layerFn(<tf.layers.Layer>this.config.pos_conv_),
                        nn.layerFn(<tf.layers.Layer>this.config.pos_conv_))(v)
                    Lp = Lp.transpose([0, 1, 4, 2, 3]).reshape([b, dim_v, dim_k, -1])
                    return tf.mul(
                        q.reshape([b, heads, 1, dim_k, -1]),
                        Lp.reshape([b, 1, dim_v, dim_k, -1]),
                    ).sum([3])
                        .transpose([0, 3, 1, 2])
                }
                else {
                    let Lp = tf.mul(
                        (<tf.Variable>this.config.pos_emb)
                            .reshape([-1, height * width, dim_k, u])
                            .transpose([0, 2, 3, 1])
                            .reshape([1, -1, dim_k, 1, u, height * width]),
                        v.reshape([b, u, dim_v, height * width])
                            .transpose([0, 2, 1, 3])
                            .reshape([b, 1, 1, dim_v, u, height * width]),
                    ).sum([4, 5])
                        .transpose([0, 3, 1, 2])

                    return tf.mul(
                        q.reshape([b, heads, dim_k, -1])
                            .transpose([0, 3, 1, 2])
                            .reshape([b, -1, heads, dim_k, 1]),
                        Lp.reshape([b, -1, 1, dim_k, dim_v]),
                    ).sum([3])
                }
            })()
            const Y = tf.add(Yc, Yp)

            const out = Y.reshape([b, height, width, heads * dim_v])
            return out
        });
    }

    /**
     * The static className getter is required by the 
     * registration step (see below).
     */
    static get className() {
        return 'LambdaLayer';
    }
}
/**
 * Regsiter the custom layer, so TensorFlow.js knows what class constructor
 * to call when deserializing an saved instance of the custom layer.
 */
tf.serialization.registerClass(LambdaLayer);
let lamlayer = new LambdaLayer({
    dim: 32,
    dim_out: 32,
    r: 3,
    dim_k: 16,
    heads: 4,
    dim_u: 4
})
let conv = tf.layers.separableConv2d({ filters: 32, kernelSize: 3 })
let input1 = tf.ones([1, 64, 64, 32])
let input2 = tf.mul(input1, 2)
let input3 = tf.mul(input2, 2)
tf.time(() => lamlayer.apply(input1))
    .then(time => console.log(`lambda kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))
    .then(() => tf.time(() => conv.apply(input1)))
    .then(time => console.log(`conv kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))
    .then(() => tf.time(() => conv.apply(input2)))
    .then(time => console.log(`lambda kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))
    .then(() => tf.time(() => conv.apply(input2)))
    .then(time => console.log(`conv kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))
    .then(() => tf.time(() => conv.apply(input3)))
    .then(time => console.log(`lambda kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))
    .then(() => tf.time(() => conv.apply(input3)))
    .then(time => console.log(`conv kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`))