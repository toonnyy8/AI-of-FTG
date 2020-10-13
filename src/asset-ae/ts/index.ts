import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

import * as tf from "@tensorflow/tfjs"
import { AED } from "../model/ae"

import { registerTfex } from "../../lib/tfjs-extensions/src/"
const tfex = registerTfex(tf)
window["train"] = false
window["autoP1"] = true
window["autoP2"] = true
window["storeMemory"] = true
console.log(`window["train"] = ${window["train"]}`)
console.log(`window["autoP1"] = ${window["autoP1"]}`)
console.log(`window["autoP2"] = ${window["autoP2"]}`)
console.log(`window["storeMemory"] = ${window["storeMemory"]}`)
const keySets: [
    {
        jump: string
        squat: string
        left: string
        right: string
        attack: {
            light: string
            medium: string
            heavy: string
        }
    },
    {
        jump: string
        squat: string
        left: string
        right: string
        attack: {
            light: string
            medium: string
            heavy: string
        }
    }
] = [
    {
        jump: "w",
        squat: "s",
        left: "a",
        right: "d",
        attack: {
            light: "j",
            medium: "k",
            heavy: "l",
        },
    },
    {
        jump: "ArrowUp",
        squat: "ArrowDown",
        left: "ArrowLeft",
        right: "ArrowRight",
        attack: {
            light: "1",
            medium: "2",
            heavy: "3",
        },
    },
]

let canvas = <HTMLCanvasElement>document.getElementById("bobylonCanvas")

let pixcanvas = <HTMLCanvasElement>document.getElementById("pixCanvas")
let d1canvas = <HTMLCanvasElement>document.getElementById("d1Canvas")

let control = (ctrl, keySet) => {
    switch (ctrl) {
        case 0: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["jump"],
                })
            )
            break
        }
        case 1: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["squat"],
                })
            )
            break
        }
        case 2: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["left"],
                })
            )
            break
        }
        case 3: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["right"],
                })
            )
            break
        }
        case 4: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["attack"]["light"],
                })
            )
            break
        }
        case 5: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["attack"]["medium"],
                })
            )
            break
        }
        case 6: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["attack"]["heavy"],
                })
            )
            break
        }
        case 7: {
            break
        }
        case 8: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["jump"],
                })
            )
            if (ctrl == 7) break
        }
        case 9: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["squat"],
                })
            )
            if (ctrl == 7) break
        }
        case 10: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["left"],
                })
            )
            if (ctrl == 7) break
        }
        case 11: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["right"],
                })
            )
            if (ctrl == 7) break
        }
        case 12: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["light"],
                })
            )
            if (ctrl == 7) break
        }
        case 13: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["medium"],
                })
            )
            if (ctrl == 7) break
        }
        case 14: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["heavy"],
                })
            )
            if (ctrl == 7) break
        }
    }
}

const memory = (size) => {
    const mem: tf.Tensor[] = []
    const read = () => {
        const idx = Math.floor(Math.random() * mem.length)
        return tf.tidy(() => mem[idx])
    }
    const write = (...ms: tf.Tensor[]) => {
        ms.forEach((m) => {
            mem.push(m)
            if (mem.length > size) {
                mem.shift().dispose()
            }
        })
    }
    return {
        read,
        write,
    }
}

tf.setBackend("webgl")
    .then(() => Game(keySets, canvas))
    .then(({ next, render, getP1, getP2, getRestart }) => {
        let op = tf.train.adamax(1e-2)
        let dh: number = 4,
            dk: number = 8,
            assetSize: number = 32,
            assetNum: number = 64,
            assetGroups: number = 4,
            down: number = 4
        let [{ fn: en_fn, ws: en_ws }, { fn: asset_fn, ws: asset_ws }, { fn: de_fn, ws: de_ws }] = AED({
            dh,
            dk,
            assetSize,
            assetNum,
            assetGroups,
            down,
            // pureAED: true,
        })

        let count = 0
        let { read, write } = memory(1024)

        console.log(...en_ws(), ...asset_ws(), ...de_ws())

        document.getElementById("save").onclick = () => {
            tf.tidy(() => {
                let tList = [...en_ws(), ...asset_ws(), ...de_ws()].reduce((acc, w) => {
                    acc[w.name] = w
                    return acc
                }, {})
                let blob = new Blob([tfex.sl.save(tList)])
                let a = document.createElement("a")
                let url = window.URL.createObjectURL(blob)
                let filename = `w-${dh}h-${dk}k-${assetSize}s-${assetNum}n-${assetGroups}g.bin`
                a.href = url
                a.download = filename
                a.click()
                window.URL.revokeObjectURL(url)
            })
        }

        document.getElementById("load").onclick = () => {
            tf.tidy(() => {
                tf.tidy(() => {
                    let load = document.createElement("input")
                    load.type = "file"
                    load.accept = ".bin"

                    load.onchange = (event) => {
                        const files = load.files
                        var reader = new FileReader()
                        reader.addEventListener("loadend", () => {
                            let loadWeights = tfex.sl.load(new Uint8Array(<ArrayBuffer>reader.result))
                            ;[...en_ws(), ...asset_ws(), ...de_ws()].forEach((w) => {
                                if (loadWeights[w.name])
                                    try {
                                        w.assign(<tf.Tensor>(<unknown>loadWeights[w.name]))
                                    } catch (e) {
                                        console.error(e)
                                    }
                            })
                        })
                        reader.readAsArrayBuffer(files[0])
                    }

                    load.click()
                })
            })
        }
        const loop = () => {
            count++
            next()
            render()
            if (count % 2 == 0) {
                let ctrl1 = Math.floor(Math.random() * 15)
                let ctrl2 = Math.floor(Math.random() * 15)
                if (window["autoP1"]) control(ctrl1, keySets[0])
                if (window["autoP2"]) control(ctrl2, keySets[1])
            }
            tf.tidy(() => {
                let pix = tf.image
                    .resizeBilinear(
                        <tf.Tensor3D>tf.maxPool(<tf.Tensor3D>tf.browser.fromPixels(canvas), [15, 10], [15, 10], "same"),
                        [64, 64]
                    )
                    .cast("float32")
                    .div(255)
                tf.browser.toPixels(<tf.Tensor3D>pix, pixcanvas)
                if (window["storeMemory"])
                    write(
                        tf.keep(pix),
                        tf.keep(pix.reverse([1, 2])),
                        tf.keep(pix.reverse([1])),
                        tf.keep(pix.reverse([2]))
                    )

                if (window["train"] == false) {
                    let query = en_fn(pix.expandDims(0))
                    let z = asset_fn(query)
                    // let z = query

                    let out = de_fn(z)

                    tf.browser.toPixels(<tf.Tensor3D>out.squeeze([0]), d1canvas)
                } else if (count % 10 == 0) {
                    let b = tf.stack([pix, read(), read(), read()], 0)
                    // let z:tf.Tensor
                    op.minimize(
                        () => {
                            let query = en_fn(b)
                            let z = asset_fn(query)
                            // let z = query

                            let out = de_fn(z)

                            let gx = (inp: tf.Tensor) => {
                                let [batch, h, w, c] = inp.shape
                                return tf
                                    .conv2d(
                                        <tf.Tensor4D>inp.reshape([batch * c, h, w, 1]),
                                        tf.tensor4d([1, 0, -1], [1, 3, 1, 1]),
                                        1,
                                        "same"
                                    )
                                    .reshape([batch, h, w, c])
                                    .add(1)
                                    .div(2)
                            }
                            let gy = (inp: tf.Tensor) => {
                                let [batch, h, w, c] = inp.shape
                                return tf
                                    .conv2d(
                                        <tf.Tensor4D>inp.reshape([batch * c, h, w, 1]),
                                        tf.tensor4d([1, 0, -1], [3, 1, 1, 1]),
                                        1,
                                        "same"
                                    )
                                    .reshape([batch, h, w, c])
                                    .add(1)
                                    .div(2)
                            }

                            let loss1 = tf.losses.logLoss(b, out)
                            let loss2 = tf.losses.logLoss(gx(b), gx(out))
                            let loss3 = tf.losses.logLoss(gy(b), gy(out))
                            let loss = tf.addN([loss1.mul(0.4), loss2.mul(0.3), loss3.mul(0.3)])
                            tf.browser.toPixels(<tf.Tensor3D>out.unstack(0)[0], d1canvas)

                            loss.print()
                            return <tf.Scalar>loss
                        },
                        false,
                        [...en_ws(), ...asset_ws(), ...de_ws()]
                    )
                }
            })

            requestAnimationFrame(loop)
        }
        requestAnimationFrame(loop)
    })
