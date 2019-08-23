import * as tf from "@tensorflow/tfjs"

export class VariableScope {
    constructor(name) {
        this.scopeName = `${name == undefined ? "" : name}`

        if (Object.keys(VariableScope._scopeList).find((scopeListName) => this.scopeName == scopeListName) == undefined) {
            this._variableList = {}
            VariableScope._scopeList[this.scopeName] = this
        }

        return VariableScope._scopeList[this.scopeName]
    }

    variableScope(name) {
        return new VariableScope(`${this.scopeName}/${name}`)
    }

    getVariable(name, shape, dtype = "float32", initializer = tf.initializers.randomNormal({ mean: 0, stddev: 1, seed: 1 }), trainable = true) {
        if (Object.keys(this._variableList).find((variableListName) => name == variableListName) == undefined) {
            this._variableList[name] = tf.tidy(() => {
                return tf.variable(initializer.apply(shape, dtype), trainable, `${this.scopeName}/${name}`, dtype)
            })
        }
        return this._variableList[name]
    }

    dispose(name) {
        if (name != null) {
            this._variableList[name].dispose()
            delete this._variableList[name]
        } else {
            Object.keys(this._variableList).forEach((key) => {
                this._variableList[key].dispose()
            })
            this._variableList = {}
        }
    }

    get scopeList() {
        return JSON.parse(JSON.stringify(VariableScope._scopeList))
    }

}
VariableScope._scopeList = {}