import * as tf from "@tensorflow/tfjs"

interface SaveJson {
    name: string
    dtype: "f64" | "f32" | "i64" | "i32" | "i16" | "i8" | "u64" | "u32" | "u16" | "u8"
}

export const save = (
    inps: {
        name: string
        values:
            | Float64Array
            | Float32Array
            | BigInt64Array
            | Int32Array
            | Int16Array
            | Int8Array
            | BigUint64Array
            | Uint32Array
            | Uint16Array
            | Uint8Array
    }[]
) => {
    return new Blob([
        "myz",
        ...(inps.reduce((prev, inp) => {
            let dtype: string = ""
            if (inp.values instanceof Float64Array) {
                dtype = "f64"
            } else if (inp.values instanceof Float32Array) {
                dtype = "f32"
            } else if (inp.values instanceof BigInt64Array) {
                dtype = "i64"
            } else if (inp.values instanceof Int32Array) {
                dtype = "i32"
            } else if (inp.values instanceof Int16Array) {
                dtype = "i16"
            } else if (inp.values instanceof Int8Array) {
                dtype = "i8"
            } else if (inp.values instanceof BigUint64Array) {
                dtype = "u64"
            } else if (inp.values instanceof Uint32Array) {
                dtype = "u32"
            } else if (inp.values instanceof Uint16Array) {
                dtype = "u16"
            } else if (inp.values instanceof Uint8Array) {
                dtype = "u8"
            }
            let json = `{"name":"${inp.name}","dtype":"${dtype}"}`
            return [...prev, new Uint32Array([json.length, inp.values.buffer.byteLength]), json, inp.values]
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
        if (String.fromCharCode(...new Uint8Array(arrBuf.slice(0, 3))) == "myz") {
            let at = 3
            let outs: {
                name: string
                values:
                    | Float64Array
                    | Float32Array
                    | BigInt64Array
                    | Int32Array
                    | Int16Array
                    | Int8Array
                    | BigUint64Array
                    | Uint32Array
                    | Uint16Array
                    | Uint8Array
            }[] = []
            while (at < arrBuf.byteLength) {
                const jsonLen = new Uint32Array(arrBuf.slice(at, at + 4))[0]
                at += 4

                const bufLen = new Uint32Array(arrBuf.slice(at, at + 4))[0]
                at += 4
                const { name, dtype } = <SaveJson>(
                    JSON.parse(String.fromCharCode(...new Uint8Array(arrBuf.slice(at, at + jsonLen))))
                )
                at += jsonLen
                let values:
                    | Float64Array
                    | Float32Array
                    | BigInt64Array
                    | Int32Array
                    | Int16Array
                    | Int8Array
                    | BigUint64Array
                    | Uint32Array
                    | Uint16Array
                    | Uint8Array
                switch (dtype) {
                    case "f64":
                        values = new Float64Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "f32":
                        values = new Float32Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "i64":
                        values = new BigInt64Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "i32":
                        values = new Int32Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "i16":
                        values = new Int16Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "i8":
                        values = new Int8Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "u64":
                        values = new BigUint64Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "u32":
                        values = new Uint32Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "u16":
                        values = new Uint16Array(arrBuf.slice(at, at + bufLen))
                        break
                    case "u8":
                        values = new Uint8Array(arrBuf.slice(at, at + bufLen))
                        break
                }

                at += bufLen
                outs.push({ name, values })
            }

            return outs
        } else {
            throw new Error("input file isn't myz")
        }
    })
}
