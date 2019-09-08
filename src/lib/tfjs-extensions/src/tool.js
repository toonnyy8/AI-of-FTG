import * as tf from "@tensorflow/tfjs"

class SequenceTidy {
    constructor(startFunc = () => { }) {
        this.funcs = []
        this.funcs.push(startFunc)
    }

    next(func = () => { }) {
        this.funcs.push(func)
        return this
    }

    run(...inputs) {
        return tf.tidy(() => {
            let output = [tf.tidy(() => {
                return this.funcs[0](...inputs)
            })]
            for (let i = 1; i < this.funcs.length; i++) {
                let newOutput = [tf.tidy(() => {
                    return this.funcs[i](...output)
                })]
                tf.dispose(output)
                output = newOutput
            }
            return output
        })
    }
}

export function sequenceTidy(func = () => { }) {
    return new SequenceTidy(func)
}

export class TensorPtr {
    constructor(tensor = null) {
        if ((tensor instanceof tf.Tensor) || tensor == null) {
            this._ptr = tensor
        } else {
            console.error(`tensor must be an instance of tf.Tensor`)
            return
        }
    }
    get ptr() {
        return this._ptr
    }
    set ptr(tensor) {
        if ((tensor instanceof tf.Tensor) || tensor == null) {
            tf.dispose(this._ptr)
            this._ptr = tensor
            return this._ptr
        } else {
            console.error(`tensor must be an instance of tf.Tensor`)
            return
        }
    }
    read() {
        return this._ptr
    }
    assign(tensor) {
        if ((tensor instanceof tf.Tensor) || tensor == null) {
            tf.dispose(this._ptr)
            this._ptr = tensor
            return this
        } else {
            console.error(`tensor must be an instance of tf.Tensor`)
            return
        }
    }
    sequence(func = () => { }) {
        func(this)
        return this
    }
}

export function tensorPtr(tensor = null) {
    return new TensorPtr(tensor)
}

export class TensorPtrList {
    constructor(tensorList = {}) {
        this._ptrList = {}
        this.assign(tensorList)
    }

    assign(tensorList = {}) {
        if (Object.values(tensorList).find(tensor => !(tensor instanceof tf.Tensor)) != undefined) {
            console.error(`tensor must be of type tf.Tensor`)
            return
        }
        Object.keys(tensorList).forEach(key => {
            tf.dispose(this._ptrList[key])
            this._ptrList[key] = tensorList[key]
        })
        return this
    }

    read(key) {
        return this._ptrList[key]
    }

    reName(oldName, newName) {
        this._ptrList[newName] = this._ptrList[oldName]
        delete this._ptrList[oldName]
        return this
    }

    sequence(func = () => { }) {
        func(this)
        return this
    }
}

export function tensorPtrList(tensorList) {
    return new TensorPtrList(tensorList)
}