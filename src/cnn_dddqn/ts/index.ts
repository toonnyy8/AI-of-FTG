import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

import * as tf from "@tensorflow/tfjs"
import { AED } from "../model/ae"
import * as nn from "../model/nn"

import { registerTfex } from "../../lib/tfjs-extensions/src/"
const tfex = registerTfex(tf)

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
let d2canvas = <HTMLCanvasElement>document.getElementById("d2Canvas")
let d3canvas = <HTMLCanvasElement>document.getElementById("d3Canvas")
let d4canvas = <HTMLCanvasElement>document.getElementById("d4Canvas")
let d5canvas = <HTMLCanvasElement>document.getElementById("d5Canvas")

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
tf.setBackend("webgl")
    .then(() => Game(keySets, canvas))
    .then(({ next, getP1, getP2, getRestart }) => {
        let op = tf.train.adamax(1e-4)
        let [{ fn: ae1, ws: ae1_ws }, { fn: ad1, ws: ad1_ws }] = AED([3, 128])
        let [{ fn: ae2, ws: ae2_ws }, { fn: ad2, ws: ad2_ws }] = AED([128, 128])
        // let [{ fn: ae3, ws: ae3_ws }, { fn: ad3, ws: ad3_ws }] = AED([64, 128])
        // let [{ fn: ae4, ws: ae4_ws }, { fn: ad4, ws: ad4_ws }] = AED([128, 256])

        let count = 0

        document.getElementById("save").onclick = () => {
            tf.tidy(() => {
                let tList = [
                    ...ae1_ws(),
                    ...ae2_ws(),
                    // ...ae3_ws(),
                    // ...ae4_ws(),
                    // ...ad4_ws(),
                    // ...ad3_ws(),
                    ...ad2_ws(),
                    ...ad1_ws(),
                ].reduce((acc, w) => {
                    acc[w.name] = w
                    return acc
                }, {})
                let blob = new Blob([tfex.sl.save(tList)])
                let a = document.createElement("a")
                let url = window.URL.createObjectURL(blob)
                let filename = "w.bin"
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
                            ;[
                                ...ae1_ws(),
                                ...ae2_ws(),
                                // ...ae3_ws(),
                                // ...ae4_ws(),
                                // ...ad4_ws(),
                                // ...ad3_ws(),
                                ...ad2_ws(),
                                ...ad1_ws(),
                            ].forEach((w) => {
                                w.assign(<tf.Tensor>(<unknown>loadWeights[w.name]))
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
            if (count % 2 == 0) {
                let ctrl1 = Math.floor(Math.random() * 15)
                let ctrl2 = Math.floor(Math.random() * 15)
                control(ctrl1, keySets[0])
                control(ctrl2, keySets[1])
            }
            if (count % 10 == 0) {
                count = 0
                tf.tidy(() => {
                    op.minimize(
                        () => {
                            let pix = tf.image
                                .resizeBilinear(
                                    <tf.Tensor3D>(
                                        tf.maxPool(
                                            <tf.Tensor3D>tf.browser.fromPixels(canvas),
                                            [30, 20],
                                            [30, 20],
                                            "same"
                                        )
                                    ),
                                    // <tf.Tensor3D>tf.browser.fromPixels(canvas),
                                    [32, 32]
                                )
                                .cast("float32")
                                .div(255)
                            let b = tf.stack([pix, pix.reverse([1, 2]), pix.reverse([1]), pix.reverse([2])], 0)
                            let e1 = ae1(b)
                            let e2 = ae2(e1)
                            let e3 = ae2(e2)
                            let e4 = ae2(e3)
                            let e5 = ae2(e4)
                            let d1 = ad1(e1)
                            let d2 = ad1(nn.mish(ad2(e2)))
                            let d3 = ad1(nn.mish(ad2(nn.mish(ad2(e3)))))
                            let d4 = ad1(nn.mish(ad2(nn.mish(ad2(nn.mish(ad2(e4)))))))
                            let d5 = ad1(nn.mish(ad2(nn.mish(ad2(nn.mish(ad2(nn.mish(ad2(e5)))))))))

                            let loss1_1 = tf.losses.sigmoidCrossEntropy(b, d1)
                            let loss2_1 = tf.losses.sigmoidCrossEntropy(b, d2)
                            let loss3_1 = tf.losses.sigmoidCrossEntropy(b, d3)
                            let loss4_1 = tf.losses.sigmoidCrossEntropy(b, d4)
                            let loss5_1 = tf.losses.sigmoidCrossEntropy(b, d5)

                            tf.browser.toPixels(<tf.Tensor3D>pix, pixcanvas)
                            tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d1.unstack(0)[0]), d1canvas)
                            tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d2.unstack(0)[0]), d2canvas)
                            tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d3.unstack(0)[0]), d3canvas)
                            tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d4.unstack(0)[0]), d4canvas)
                            tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d5.unstack(0)[0]), d5canvas)
                            let loss = tf.addN([
                                loss1_1.mul(0.2),
                                loss2_1.mul(0.2),
                                loss3_1.mul(0.2),
                                loss4_1.mul(0.2),
                                loss5_1.mul(0.2),
                            ])
                            loss.print()
                            return <tf.Scalar>loss
                        },
                        false,
                        [
                            ...ae1_ws(),
                            ...ae2_ws(),
                            // ...ae3_ws(),
                            // ...ae4_ws(),
                            // ...ad4_ws(),
                            // ...ad3_ws(),
                            ...ad2_ws(),
                            ...ad1_ws(),
                        ]
                    )
                })
            }

            requestAnimationFrame(loop)
        }
        requestAnimationFrame(loop)
    })
