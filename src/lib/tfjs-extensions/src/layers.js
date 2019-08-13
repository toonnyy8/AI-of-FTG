import * as tf from "@tensorflow/tfjs"

export class LayerNormalization extends tf.layers.Layer {
    constructor(init = { axis }) {
        super({})
        this.supportsMasking = true
        this.axis = init.axis || -1
    }

    apply(inputs, kwargs) {
        let input = inputs
        if (Array.isArray(input)) {
            input = input[0]
        }
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

    /**
     * Layers must implement "getClassName".
     */
    getClassName() {
        return 'LayerNormalization'
    }
}

export function layerNormalization(init = { axis }) {
    return new LayerNormalization(init)
}


export class Lambda extends tf.layers.Layer {
    constructor({ func = () => { }, outputShape = null }) {
        super({})
        this.func = func
        this.oS = outputShape
    }

    apply(inputs, kwargs) {
        return tf.tidy(() => {
            return super.apply(inputs, kwargs)
        })
    };

    call(inputs, kwargs) {
        return tf.tidy(() => this.func(...inputs))
    };

    computeOutputShape(inputShape) {
        return this.oS || inputShape
    };
    /**
     * Layers must implement "getClassName".
     */
    getClassName() {
        return "Lambda"
    }
}

export function lambda({ func = () => { }, outputShape = null }) {
    return new Lambda({ func, outputShape })
}