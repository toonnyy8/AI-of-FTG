import * as tf from "@tensorflow/tfjs"

interface SaveJson {
    name: string
    shape: number[]
}

export const save = (
    inps: {
        name: string
        tensor: tf.Tensor
    }[]
) => {
    return new Blob([
        "imgz",
        ...(inps.reduce((prev, inp) => {
            let buf = inp.tensor.bufferSync<"int32">()
            let json = `{"name":"${inp.name}","shape":[${buf.shape}]}`
            return [
                ...prev,
                new Uint32Array([json.length, buf.values.length]),
                json,
                new Uint8Array(buf.values),
            ]
        }, [] as BlobPart[]) as BlobPart[]),
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
        if (String.fromCharCode(...new Uint8Array(arrBuf.slice(0, 4))) == "imgz") {
            let at = 4
            let outs: {
                name: string
                tensor: tf.Tensor
            }[] = []
            while (at < arrBuf.byteLength) {
                const jsonLen = new Uint32Array(arrBuf.slice(at, at + 4))[0]
                at += 4

                const bufLen = new Uint32Array(arrBuf.slice(at, at + 4))[0]
                at += 4
                const { name, shape } = <SaveJson>(
                    JSON.parse(String.fromCharCode(...new Uint8Array(arrBuf.slice(at, at + jsonLen))))
                )
                at += jsonLen

                let values: Int32Array = new Int32Array(new Uint8Array(arrBuf.slice(at, at + bufLen)))

                at += bufLen
                outs.push({ name: name, tensor: tf.tensor(values, shape, "int32") })
            }

            return outs
        } else {
            throw new Error("input file isn't imgz")
        }
    })
}
