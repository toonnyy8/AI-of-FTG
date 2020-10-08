import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

import * as tf from "@tensorflow/tfjs"
import { AED } from "../model/ae"
import * as nn from "../model/nn"

import { registerTfex } from "../../lib/tfjs-extensions/src/"
import { dot, tidy } from "@tensorflow/tfjs"
const tfex = registerTfex(tf)
window["train"] = false
console.log(`window["train"] = ${window["train"]}`)
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
        let dk = 4
        let dv = 64
        let bookSize = 256
        let inpLayer = tf.layers.separableConv2d({
            kernelSize: 3,
            filters: dk,
            padding: "same",
            activation: "selu",
            trainable: true,
        })
        inpLayer.build([1, 1, 1, 3])
        let encoder = tf.sequential({
            layers: [
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dk,
                    padding: "same",
                    activation: "selu",
                    inputShape: [1, 1, dk],
                    trainable: true,
                }),
                // tf.layers.separableConv2d({
                //     kernelSize: 3,
                //     filters: dk,
                //     padding: "same",
                //     // dilationRate: 2,
                //     activation: "selu",
                //     trainable: true,
                // }),
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dk,
                    padding: "same",
                    strides: 2,
                    activation: "selu",
                    trainable: true,
                }),
            ],
        })
        encoder.build([1, 1, 1, dk])

        let decoder = tf.sequential({
            layers: [
                tf.layers.upSampling2d({ size: [2, 2], inputShape: [1, 1, dv] }),
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dv,
                    padding: "same",
                    activation: "selu",
                    trainable: true,
                }),
                tf.layers.separableConv2d({
                    kernelSize: 3,
                    filters: dv,
                    padding: "same",
                    // dilationRate: 2,
                    activation: "selu",
                    trainable: true,
                }),
                // tf.layers.separableConv2d({
                //     kernelSize: 3,
                //     filters: dv / 2,
                //     padding: "same",
                //     // dilationRate: 4,
                //     activation: "selu",
                //     trainable: true,
                // }),
            ],
        })
        decoder.build([1, 1, 1, dv])

        let outLayer = tf.layers.separableConv2d({
            kernelSize: 3,
            filters: 3,
            padding: "same",
            trainable: true,
        })
        outLayer.build([1, 1, 1, dv])

        let bookLayer = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    inputShape: [1, 1, dk],
                    filters: bookSize,
                    kernelSize: 1,
                    useBias: false,
                    activation: "softmax",
                }),
                tf.layers.conv2d({
                    filters: dv,
                    kernelSize: 1,
                    useBias: false,
                })
            ]
        })

        let count = 0

        document.getElementById("save").onclick = () => {
            tf.tidy(() => {
                let tList = [
                    ...(<tf.Variable[]>inpLayer.getWeights()),
                    ...(<tf.Variable[]>encoder.getWeights()),
                    ...(<tf.Variable[]>decoder.getWeights()),
                    ...(<tf.Variable[]>outLayer.getWeights()),
                    ...(<tf.Variable[]>bookLayer.getWeights()),
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
                                    ...(<tf.Variable[]>inpLayer.getWeights()),
                                    ...(<tf.Variable[]>encoder.getWeights()),
                                    ...(<tf.Variable[]>decoder.getWeights()),
                                    ...(<tf.Variable[]>outLayer.getWeights()),
                                    ...(<tf.Variable[]>bookLayer.getWeights()),
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
            if (window["train"] == false) {
                tf.tidy(() => {
                    let pix = tf.image
                        .resizeBilinear(
                            <tf.Tensor3D>(
                                tf.maxPool(<tf.Tensor3D>tf.browser.fromPixels(canvas), [30, 20], [30, 20], "same")
                            ),
                            [64, 64]
                        )
                        .cast("float32")
                        .div(255)
                    tf.browser.toPixels(<tf.Tensor3D>pix, pixcanvas)

                    let query = <tf.Tensor>(
                        encoder.apply(
                            encoder.apply(encoder.apply(encoder.apply(inpLayer.apply(pix.expandDims(0)))))
                        )
                    )
                    let z = bookLayer.apply(query)
                    // let z = query

                    let out = <tf.Tensor>(
                        outLayer.apply(decoder.apply(decoder.apply(decoder.apply(decoder.apply(z)))))
                    )

                    tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>out.squeeze([0])), d1canvas)
                })
            }
            else if (count % 10 == 0) {
                count = 0
                tf.tidy(() => {
                    let pix = tf.image
                        .resizeBilinear(
                            <tf.Tensor3D>(
                                tf.maxPool(<tf.Tensor3D>tf.browser.fromPixels(canvas), [15, 10], [15, 10], "same")
                                // tf.browser.fromPixels(canvas)
                            ),
                            [64, 64]
                        )
                        .cast("float32")
                        .div(255)
                    tf.browser.toPixels(<tf.Tensor3D>pix, pixcanvas)
                    let b = tf.stack([pix, pix.reverse([1, 2]), pix.reverse([1]), pix.reverse([2])], 0)
                    // let z:tf.Tensor
                    op.minimize(
                        () => {
                            let query = <tf.Tensor>(
                                encoder.apply(encoder.apply(encoder.apply(encoder.apply(inpLayer.apply(b)))))
                            )
                            let z = bookLayer.apply(query)
                            // let z = query

                            let out = <tf.Tensor>(
                                outLayer.apply(decoder.apply(decoder.apply(decoder.apply(decoder.apply(z)))))
                            )

                            let loss = tf.losses.sigmoidCrossEntropy(b, out)
                            tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>out.unstack(0)[0]), d1canvas)

                            loss.print()
                            return <tf.Scalar>loss
                        },
                        false,
                        [
                            ...(<tf.Variable[]>inpLayer.getWeights()),
                            ...(<tf.Variable[]>encoder.getWeights()),
                            ...(<tf.Variable[]>decoder.getWeights()),
                            ...(<tf.Variable[]>outLayer.getWeights()),
                            ...(<tf.Variable[]>bookLayer.getWeights()),
                        ]
                    )
                })
            }

            requestAnimationFrame(loop)
        }
        requestAnimationFrame(loop)
    })
