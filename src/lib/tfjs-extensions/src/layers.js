import * as tf from "@tensorflow/tfjs"

// https://github.com/GistNoesis/Wisteria/blob/master/tfjs/src/LayerNorm.js
export class LayerNormalization extends tf.layers.Layer {
    constructor(args) {
        super({})
        this.axis = args.axis
        //console.log("LayerNorm constructor")
        //console.log(this.axis)
    }
    computeCorrectName(name) {
        //layer_norm_LayerNorm1/g  must be converted to layer_norm/g
        //layer_norm_LayerNorm2/g  must be converted to layer_norm_1/g
        let prefix = "layer_norm_LayerNorm"
        let subsr = name.substr(prefix.length)
        let vals = subsr.split('/')
        let num = parseInt(vals[0])
        let nbstr = num == 1 ? "" : "_" + (num - 1).toString()
        let res = "layer_norm" + nbstr + "/" + vals[1]
        return res
    }
    build(inputShape) {
        // console.log("LayerNorm build : ")
        this.g = this.addWeight("g", [inputShape[inputShape.length - 1]], "float32", tf.initializers.ones())
        this.b = this.addWeight("b", [inputShape[inputShape.length - 1]], "float32", tf.initializers.zeros())
        let gname = this.computeCorrectName(this.g.originalName)
        let bname = this.computeCorrectName(this.b.originalName)
        this.g.originalName = gname
        this.g.name = gname
        this.b.originalName = bname
        this.b.name = bname
        // console.log(this.g)
        // console.log(this.b)
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
        return "LayerNormalization"
    }
}

export function layerNormalization(init = { axis }) {
    return new LayerNormalization(init)
}
// registerClass
tf.serialization.registerClass(LayerNormalization)



export class Lambda extends tf.layers.Layer {
    constructor({ func = () => { } }) {
        super({})
        this.func = func
    }

    apply(inputs, kwargs) {
        return tf.tidy(() => {
            return super.apply(inputs, kwargs)
        })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => this.func(...inputs))
    }

    computeOutputShape(inputShape) {
        return tf.tidy(() => {
            let tempInput
            let tempOutput
            if (inputShape[0] != null) {
                tempInput = inputShape.map((shape) => {
                    shape.shift()
                    return tf.ones(shape)
                })
                tempOutput = this.func(...tempInput)
            } else {
                inputShape.shift()
                tempInput = tf.ones(inputShape)
                tempOutput = this.func(tempInput)
            }

            if (tempOutput instanceof tf.Tensor) {
                return [null].concat(tempOutput.shape)
            } else {
                return tempOutput.map((t) => {
                    return [null].concat(t.shape)
                })
            }

        })
    }

    /*
    * If a custom layer class is to support serialization, it must implement
    * the `className` static getter.
    */
    static get className() {
        return "Lambda"
    }
}

export function lambda({ func = () => { } }) {
    return new Lambda({ func })
}

// registerClass
tf.serialization.registerClass(Lambda)