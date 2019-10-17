import * as tf from "@tensorflow/tfjs"

export function registerTool(tf_ = tf) {

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
            return tf_.tidy(() => {
                let output = [tf_.tidy(() => {
                    return this.funcs[0](...inputs)
                })]
                for (let i = 1; i < this.funcs.length; i++) {
                    let newOutput = [tf_.tidy(() => {
                        return this.funcs[i](...output)
                    })]
                    tf_.dispose(output)
                    output = newOutput
                }
                return output
            })
        }
    }

    function sequenceTidy(func = () => { }) {
        return new SequenceTidy(func)
    }

    class TensorPtr {
        constructor(tensor = null) {
            if ((tensor instanceof tf_.Tensor) || tensor == null) {
                this._ptr = tensor
            } else {
                console.error(`tensor must be an instance of tf_.Tensor`)
                return
            }
        }
        get ptr() {
            return this._ptr
        }
        set ptr(tensor) {
            if ((tensor instanceof tf_.Tensor) || tensor == null) {
                tf_.dispose(this._ptr)
                this._ptr = tensor
                return this._ptr
            } else {
                console.error(`tensor must be an instance of tf_.Tensor`)
                return
            }
        }
        read() {
            return this._ptr
        }
        assign(tensor) {
            if ((tensor instanceof tf_.Tensor) || tensor == null) {
                tf_.dispose(this._ptr)
                this._ptr = tensor
                return this
            } else {
                console.error(`tensor must be an instance of tf_.Tensor`)
                return
            }
        }
        sequence(func = () => { }) {
            func(this)
            return this
        }
    }

    function tensorPtr(tensor = null) {
        return new TensorPtr(tensor)
    }

    class TensorPtrList {
        constructor(tensorList = null) {
            this._ptrList = {}
            if (tensorList !== null) {
                this.assign(tensorList)
            }
        }

        assign(tensorList = {}) {
            if (Object.values(tensorList).find(tensor => !(tensor instanceof tf_.Tensor)) != undefined) {
                console.error(`tensor must be of type tf_.Tensor`)
                return
            }
            Object.keys(tensorList).forEach(key => {
                tf_.dispose(this._ptrList[key])
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

    function tensorPtrList(tensorList) {
        return new TensorPtrList(tensorList)
    }

    return {
        sequenceTidy,
        TensorPtr,
        tensorPtr,
        TensorPtrList,
        tensorPtrList
    }
}