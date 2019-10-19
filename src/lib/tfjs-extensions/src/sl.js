import * as tf from "@tensorflow/tfjs"
import * as binary from "js-binary"
import { default as toBuffer } from "typedarray-to-buffer"

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


export function registerSL(tf_ = tf) {
    function save(tList) {
        let save_ = (t) => {
            return tf_.tidy(() => {
                if (t instanceof tf_.Tensor) {
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
                    console.error(`tensor must be an instance of tf_.Tensor`)
                }
            })
        }

        return saverCoder.encode({
            keys: Object.keys(tList),
            tensors: Object.values(tList).map(t => save_(t))
        })
    }

    function load(saver) {
        return tf_.tidy(() => {
            let tList = saverCoder.decode(toBuffer(saver))
            return tList.keys.reduce((last, key, idx) => {
                let tObj = tensorCoder.decode(tList.tensors[idx])
                tObj.values = dtypeCoder[tObj.dtype].decode(tObj.values)
                last[key] = tf_.tensor(tObj.values, tObj.shape, tObj.dtype)
                return last
            }, {})
        })
    }

    return { save, load }
}