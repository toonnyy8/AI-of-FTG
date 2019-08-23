import * as tf from "@tensorflow/tfjs"

export declare class VariableScope {
    scopeName: String
    scopeList: Object
    variableList:Object
    constructor(name: String)
    variableScope(name: String): VariableScope
    getVariable(
        name: String, shape?: Number[], dtype?: "float32" | "int32" | "bool" | "complex64" | "string", initializer ?:tf.serialization.Serializable, trainable ?: Boolean
        ): tf.Variable
    dispose(name?: String): void
}

export declare function variableScope(name: String): VariableScope 