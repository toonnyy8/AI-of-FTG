import * as tf from "@tensorflow/tfjs"

export function registerLayers(tf_ = tf) {

    // https://github.com/GistNoesis/Wisteria/blob/master/tfjs/src/LayerNorm.js
    class LayerNormalization extends tf_.layers.Layer {
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
            this.g = this.addWeight("g", [inputShape[inputShape.length - 1]], "float32", tf_.initializers.ones())
            this.b = this.addWeight("b", [inputShape[inputShape.length - 1]], "float32", tf_.initializers.zeros())
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
            let u = tf_.mean(x, axis, true)
            let xmu = tf_.sub(x, u)
            let s = tf_.mean(tf_.square(xmu), axis, true)
            x = tf_.mul(xmu, tf_.rsqrt(tf_.add(s, epsilon)))
            //let gval = this.g.read()
            let gval = tf_.reshape(this.g.read(), [1, 1, -1])
            let bval = tf_.reshape(this.b.read(), [1, 1, -1])
            //x = x * + tf_.reshape( this.b,[1,1,-1])
            x = tf_.add(tf_.mul(x, gval), bval)
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

    function layerNormalization(init = { axis }) {
        return new LayerNormalization(init)
    }
    // registerClass
    tf_.serialization.registerClass(LayerNormalization)



    class Lambda extends tf_.layers.Layer {
        constructor({ func = () => { }, outputShape = null }) {
            super({})
            this.func = func
            this.customOutputShape = outputShape
        }

        apply(inputs, kwargs) {
            return tf_.tidy(() => {
                return super.apply(inputs, kwargs)
            })
        }

        call(inputs, kwargs) {
            return tf_.tidy(() => this.func(...inputs))
        }

        computeOutputShape(inputShape) {
            if (this.customOutputShape == null) {
                if (inputShape[0] instanceof Array) {
                    return inputShape[0]
                } else {
                    return inputShape
                }
            } else {
                return [null].concat(this.customOutputShape)
            }
        }

        /*
        * If a custom layer class is to support serialization, it must implement
        * the `className` static getter.
        */
        static get className() {
            return "Lambda"
        }
    }

    function lambda({ func = () => { }, outputShape = null }) {
        return new Lambda({ func, outputShape })
    }

    // registerClass
    tf_.serialization.registerClass(Lambda)

    /********************************/
    return {
        LayerNormalization,
        layerNormalization,
        Lambda,
        lambda
    }
}