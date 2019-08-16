import * as tf from "@tensorflow/tfjs"

export class positionwiseFF extends tf.layers.Layer {
    constructor(args) {
        super({})
        this.axis = args.axis
        //console.log("LayerNorm constructor")
        //console.log(this.axis)
    }

    build(inputShape) {
        //console.log("LayerNorm build : ")
        this.setWeights()
        this.g = this.addWeight("g", [inputShape[inputShape.length - 1]], "float32", tf.initializers.ones())
        this.b = this.addWeight("b", [inputShape[inputShape.length - 1]], "float32", tf.initializers.zeros())
        console.log(this.getWeights())
        console.log(this.b)
        this.built = true
    }

    computeOutputShape(inputShape) {
        //console.log("LayerNorm computeOutputShape")
        //console.log(inputShape)
        return inputShape
    }

    call(inputs, kwargs) {
        let input = inputs
        if (Array.isArray(input)) {
            input = input[0]
        }
        //console.log("LayerNorm call")
        this.invokeCallHook(inputs, kwargs)
        let x = input
        let epsilon = 1e-5
        let axis = this.axis == -1 ? input.shape.length - 1 : this.axis
        let u = tf.mean(x, axis, true)
        let xmu = tf.sub(x, u)
        let s = tf.mean(tf.square(xmu), axis, true)
        x = tf.mul(xmu, tf.rsqrt(tf.add(s, epsilon)))
        //let gval = this.g.read()
        let gval = tf.reshape(this.g.read(), [1, 1, -1])
        let bval = tf.reshape(this.b.read(), [1, 1, -1])
        //x = x * + tf.reshape( this.b,[1,1,-1])
        x = tf.add(tf.mul(x, gval), bval)
        return x
    }

    /*
    * If a custom layer class is to support serialization, it must implement
    * the `className` static getter.
    */
    static get className() {
        return "positionwiseFF"
    }
}
tf.serialization.registerClass(positionwiseFF)
// export class TransformerXL {
//     constructor({ }) {
//         super()
//     }

// }

// export function transformerXL() {
//     return new TransformerXL()
// }

// function positionalEmbedding() {

// }
