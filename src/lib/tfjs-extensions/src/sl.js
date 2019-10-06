import * as tf from "@tensorflow/tfjs"
import * as binary from "js-binary"

let dtypeCoder = {
    int32: new binary.Type(['int']),
    float32: new binary.Type(['float']),
    string: new binary.Type(['string']),
    bool: new binary.Type(['boolean']),
    complex64: new binary.Type(['float'])

}

let tensorCoder = new binary.Type({
    shape: ['uint'],
    dtype: 'string',
    values: "Buffer",
})

let saverCoder = new binary.Type({
    keys: ["string"],
    tensors: ["Buffer"]
})

export function save(tList) {
    let save_ = (t) => {
        return tf.tidy(() => {
            if (t instanceof tf.Tensor) {
                let values = t.reshape([-1]).arraySync()
                if (t.dtype == "bool") {
                    values = values.map(v => v ? true : false)
                }
                return tensorCoder.encode({
                    shape: t.shape,
                    dtype: t.dtype,
                    values: dtypeCoder[t.dtype].encode(values)
                })
            } else {
                console.error(`tensor must be an instance of tf.Tensor`)
            }
        })
    }

    return saverCoder.encode({
        keys: Object.keys(tList),
        tensors: Object.values(tList).map(t => save_(t))
    })
}

export function load(saver) {
    return tf.tidy(() => {
        let tList = saverCoder.decode(saver)
        return tList.keys.reduce((last, key, idx) => {
            let tObj = tensorCoder.decode(tList.tensors[idx])
            tObj.values = dtypeCoder[tObj.dtype].decode(tObj.values)
            last[key] = tf.tensor(tObj.values, tObj.shape, tObj.dtype)
            return last
        }, {})

    })
}