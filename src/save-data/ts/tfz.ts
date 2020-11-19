import * as tf from "@tensorflow/tfjs"
import JSZip from "jszip"

interface SaveJson {
    name: string
    shape: number[]
    dtype: "float32" | "int32" | "bool"
}

export const save = (
    inps: {
        name: string
        tensor: tf.Tensor
    }[]
) => {
    return new Blob([
        "tfz",
        ...(inps.map((inp) => {
            let buf = inp.tensor.bufferSync()
            let json = `{"name":"${inp.name}","shape":${buf.shape},"dtype":"${buf.dtype}"}`
            return <unknown>[
                new Uint32Array([json.length, buf.values.length * buf.values.BYTES_PER_ELEMENT]),
                json,
                buf.values,
            ]
        }) as BlobPart[]),
    ])
}

export const load = (tfzFile: Blob | ArrayBuffer) => {
    return new Promise((resolve: (value: ArrayBuffer) => unknown, reject) => {
        if (tfzFile instanceof Blob) {
            const reader = new FileReader()
            reader.addEventListener("load", function () {
                resolve(<ArrayBuffer>reader.result)
            })
            reader.readAsArrayBuffer(tfzFile)
        } else {
            resolve(<ArrayBuffer>tfzFile)
        }
    }).then((arrBuf: ArrayBuffer) => {
        if (String.fromCharCode(...new Uint8Array(arrBuf.slice(0, 3))) == "tfz") {
            let at = 3
            let outs: {
                name: string
                tensor: tf.Tensor
            }[] = []
            while (at < arrBuf.byteLength) {
                const jsonLen = new Uint32Array(arrBuf.slice(at, at + 4))[0]
                at += 4

                const bufLen = new Uint32Array(arrBuf.slice(at, at + 4))[0]
                at += 4

                const { name, dtype, shape } = <SaveJson>(
                    JSON.parse(String.fromCharCode(...new Uint8Array(arrBuf.slice(at, at + jsonLen))))
                )
                at += jsonLen

                let values: Float32Array | Int32Array | Uint8Array
                switch (dtype) {
                    case "float32":
                        values = new Float32Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "int32":
                        values = new Int32Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "bool":
                        values = new Uint8Array(arrBuf.slice(at, at + bufLen))
                        break
                }
                at += bufLen
                outs.push({ name: name, tensor: tf.tensor(values, shape, dtype) })
            }

            return outs
        } else {
            throw new Error("input file isn't tfz")
        }
    })
}
