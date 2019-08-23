import * as tf from "@tensorflow/tfjs"
import { runInThisContext } from "vm";

export class VariableScope {
    constructor(name, path) {
        this.scopeName = `${name == undefined ? "" : name}`
        this._scopes = {}
        this._variables = {}
    }

    variableScope(name) {
        if (Object.keys(this._scopes).find((scopeName) => name == scopeName) == undefined) {
            this._scopes[name] = new VariableScope(`${this.scopeName}/${name}`)
        }
        return this._scopes[name]
    }

    getVariable(name, shape, dtype = "float32", initializer = tf.initializers.randomNormal({ mean: 0, stddev: 1, seed: 1 }), trainable = true) {
        dtype = dtype != null && dtype != undefined ? dtype : "float32"
        initializer = initializer != null && initializer != undefined ? initializer : tf.initializers.randomNormal({ mean: 0, stddev: 1, seed: 1 })
        trainable = trainable != null && trainable != undefined ? trainable : true
        if (Object.keys(this._variables).find((variablesName) => name == variablesName) == undefined) {
            this._variables[name] = tf.tidy(() => {
                return tf.variable(initializer.apply(shape, dtype), trainable, `${this.scopeName}/${name}`, dtype)
            })
        }
        return this._variables[name]
    }

    dispose(name) {
        if (name != null) {
            this._variables[name].dispose()
            delete this._variables[name]
        } else {
            Object.keys(this._variables).forEach((key) => {
                this._variables[key].dispose()
                delete this._variables[key]
            })
        }
    }

    get scopes() {
        return this._scopes
    }

    get variables() {
        return this._variables
    }

    save() {
        return tf.tidy(() => {
            return {
                variables: Object.keys(this._variables).reduce((variables, key) => {
                    variables[key] = {
                        dtype: this._variables[key].dtype,
                        shape: this._variables[key].shape,
                        trainable: this._variables[key].trainable,
                        values: this._variables[key].arraySync()
                    }
                    return variables
                }, {}),
                scopes: Object.keys(this._scopes).reduce((scopes, key) => {
                    scopes[key] = this._scopes[key].save()
                    return scopes
                }, {})
            }
        })
    }

    load(saveData) {
        return tf.tidy(() => {
            Object.keys(saveData.variables).forEach((key) => {
                tf.tidy(() => {
                    let v = this.getVariable(key, saveData.variables[key].shape, saveData.variables[key].dtype, undefined, saveData.variables[key].trainable)
                    v.assign(tf.tensor(saveData.variables[key].values, saveData.variables[key].shape, saveData.variables[key].dtype))
                })
            })
            Object.keys(saveData.scopes).forEach((key) => {
                tf.tidy(() => {
                    this.variableScope(key).load(saveData.scopes[key])
                })
            })
        })
    }
}