import * as tf from "@tensorflow/tfjs"

declare class SequenceTidy {
    funcs: [() => {}]
    constructor(func: () => {})
    next(func: () => {}): SequenceTidy
    run(input: any): tf.Tensor
}

declare class TensorPtr {
    _ptr: tf.Tensor
    ptr: tf.Tensor
    constructor(tensor: tf.Tensor)
    read(): tf.Tensor
    assign(tensor: tf.Tensor): TensorPtr
    sequence(func: (tensorPrt: TensorPtr) => {}): TensorPtr
}

declare class TensorPtrList {
    _ptrList: { [key: string]: tf.Tensor }
    constructor(tensorList: { [key: string]: tf.Tensor })
    read(key: String): tf.Tensor
    assign(tensorList: { [key: string]: tf.Tensor }): TensorPtrList
    sequence(func: (tensorPrtList: TensorPtrList) => {}): TensorPtrList
}

export declare function registerTool(tf): {
    sequenceTidy(func: () => {}): SequenceTidy
    TensorPtr: TensorPtr
    tensorPtr(tensor: tf.Tensor): TensorPtr
    TensorPtrList: TensorPtrList
    tensorPtrList(tensorList: { [key: string]: tf.Tensor }): TensorPtrList
}