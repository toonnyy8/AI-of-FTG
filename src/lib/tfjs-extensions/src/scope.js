import * as tf from "@tensorflow/tfjs"
import * as sl from "./sl"

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
            return this.allVariables().reduce((last, variable) => {
                last[variable.name.slice(this.scopeName.length)] = variable.bufferSync().toTensor()
                return last
            }, {})
        })
    }

    load(tList) {
        return tf.tidy(() => {
            return Object.keys(tList).reduce((last, scopeName) => {
                let scopeNames = scopeName.split("/")
                let variableName = scopeNames.pop()
                let scope = this
                for (let i = 1; i < scopeNames.length; i++) {
                    scope = scope.variableScope(scopeNames[i])
                }
                let v = scope.getVariable(variableName, tList[scopeName].shape, tList[scopeName].dtype)
                v.assign(tList[scopeName])
                last[v.name] = v
                return last
            }, {})
        })
    }

    trainableVariables() {
        return tf.tidy(() => {
            return Object.keys(this.variables)
                .map(key => this.variables[key])
                .filter(variable => variable.trainable == true)
                .concat([
                    Object.keys(this.scopes)
                        .map(key => this.scopes[key].trainableVariables())
                        .flat()
                ]).flat()
        })
    }

    allVariables() {
        return tf.tidy(() => {
            return Object.keys(this.variables)
                .map(key => this.variables[key])
                .concat([
                    Object.keys(this.scopes)
                        .map(key => this.scopes[key].allVariables())
                        .flat()
                ]).flat()
        })
    }
}