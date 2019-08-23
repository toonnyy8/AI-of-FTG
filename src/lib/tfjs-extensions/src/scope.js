import * as tf from "@tensorflow/tfjs"
import { runInThisContext } from "vm";

export class VariableScope {
    constructor(name, path) {
        this.scopeName = `${name == undefined ? "" : name}`
        this._scopeList = {}
        this._variableList = {}
    }

    variableScope(name) {
        if (Object.keys(this._scopeList).find((scopeName) => name == scopeName) == undefined) {
            this._scopeList[name] = new VariableScope(`${this.scopeName}/${name}`)
        }
        return this._scopeList[name]
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
                delete this._variableList[key]
            })
        }
    }

    get scopeList() {
        return this._scopeList
    }

    get variableList() {
        return this._variableList
    }
}