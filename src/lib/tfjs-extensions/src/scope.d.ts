import * as tf from "@tensorflow/tfjs"

export declare class VariableScope {
    scopeName: String
    scopes: Object
    variables: Object
    constructor(name: String)
    variableScope(name: String): VariableScope
    getVariable(
        name: String, shape?: Number[], dtype?: "float32" | "int32" | "bool" | "complex64" | "string", initializer?: tf.serialization.Serializable, trainable?: Boolean
    ): tf.Variable
    dispose(name?: String): void
    save(): Uint8Array
    load(saveData: Uint8Array): { [key: string]: tf.Variable }
    trainableVariables(): tf.Variable[]
    allVariables(): tf.Variable[]
}