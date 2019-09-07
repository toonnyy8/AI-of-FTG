import * as tf from "@tensorflow/tfjs"

declare class SequenceTidy {
    funcs: [() => {}]
    constructor(func: () => {})
    next(func: () => {}): SequenceTidy
    run(input: any): tf.Tensor
}

export declare function sequenceTidy(func: () => {}): SequenceTidy

export declare class TensorPtr {
    _ptr: tf.Tensor
    constructor(tensor: tf.Tensor)
    read(): tf.Tensor
    assign(tensor: tf.Tensor): tf.Tensor
}

export declare function tensorPtr(tensor: tf.Tensor): TensorPtr