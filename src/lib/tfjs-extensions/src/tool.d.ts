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
    ptr: tf.Tensor
    constructor(tensor: tf.Tensor)
    read(): tf.Tensor
    assign(tensor: tf.Tensor): TensorPtr
    sequence(func: (tensorPrt: TensorPtr) => {}): TensorPtr
}

export declare function tensorPtr(tensor: tf.Tensor): TensorPtr

export declare class TensorPtrList {
    _ptrList: { key: tf.Tensor }
    constructor(tensorList: { key: tf.Tensor })
    read(key: String): tf.Tensor
    assign(tensorList: { key: tf.Tensor }): TensorPtrList
    sequence(func: (tensorPrtList: TensorPtrList) => {}): TensorPtrList
}

export declare function tensorPtrList(tensorList: { key: tf.Tensor }): TensorPtrList