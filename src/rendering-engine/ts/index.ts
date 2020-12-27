import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as myz from "./myz"
import { QVAE } from "./qvae"
import { MHA, FF } from "./mha"

let canvas = <HTMLCanvasElement>document.getElementById("canvas")

tf.setBackend("webgl").then(() => {
    let dk = 8
    let {
        enc: { fn: enc_fn, ws: enc_ws },
        dec: { fn: dec_fn, ws: dec_ws },
    } = QVAE({})
    let count = 0
    let trainDatas: tf.Tensor4D[] = []

    ;(<HTMLButtonElement>document.getElementById("load-data")).onclick = () => {
        let load = document.createElement("input")
        load.type = "file"
        load.accept = ".myz"
        load.multiple = true

        load.onchange = (event) => {
            const files = Array.from(<FileList>load.files)
            trainDatas.forEach((trainData) => trainData.dispose())
            trainDatas = []

            const readLoop = (file: File, files: File[]) => {
                myz.load(file)
                    .then((datas) => {
                        tf.tidy(() => {
                            let frames = datas.find((data) => {
                                return data.name == "frames"
                            })
                            let frameShape = datas.find((data) => {
                                return data.name == "frameShape"
                            })
                            trainDatas = [
                                ...trainDatas,
                                tf.keep(
                                    tf
                                        .tensor4d(
                                            <Uint8Array>frames?.values,
                                            <[number, number, number, number]>Array.from(<Uint8Array>frameShape?.values)
                                        )
                                        .cast("float32")
                                        .div(255)
                                ),
                            ]
                        })
                    })
                    .then(() => {
                        if (files.length >= 1) {
                            readLoop(files[0], files.slice(1))
                        } else {
                            console.log(trainDatas)
                            console.log("end reading")
                        }
                    })
            }

            console.log("start reading")
            readLoop(files[0], files.slice(1))
        }

        load.click()
    }

    const opt = tf.train.adamax(0.01)
    ;(<HTMLButtonElement>document.getElementById("train-log")).onclick = async () => {
        let fileIdx = Math.floor(Math.random() * trainDatas.length)
        let batchSize = 8
        let batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let batch = <tf.Tensor4D>trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    batch = tf.concat([batch, batch.reverse(2), batch.reverse(3), batch.reverse([2, 3])])
                    let z: tf.Tensor2D,
                        q_z: tf.Tensor2D = tf.tensor2d([[0]])
                    const grads1 = opt.computeGradients(
                        () => {
                            ;({ z, q_z } = enc_fn(batch))
                            q_z = tf.keep(q_z)
                            let out = <tf.Tensor4D>dec_fn(z)
                            let mainLoss = tf.losses.logLoss(batch, out)
                            let l2Loss = [...enc_ws()]
                                .reduce((loss, w) => loss.add(w.square().mean()), tf.scalar(0))
                                .div([...enc_ws()].length)

                            return mainLoss.add(l2Loss)
                        },
                        <tf.Variable[]>(<unknown>[...enc_ws()])
                    ).grads
                    const grads2 = opt.computeGradients(
                        () => {
                            let out = <tf.Tensor4D>dec_fn(q_z)
                            let mainLoss = tf.losses.logLoss(batch, out)
                            mainLoss.print()
                            let l2Loss = [...dec_ws()]
                                .reduce((loss, w) => loss.add(w.square().mean()), tf.scalar(0))
                                .div([...dec_ws()].length)

                            return mainLoss.add(l2Loss)
                        },
                        <tf.Variable[]>(<unknown>[...dec_ws()])
                    ).grads

                    opt.applyGradients(<[]>(<unknown>{ ...grads1, ...grads2 }))
                    q_z.dispose()
                })

                fileIdx = Math.floor(Math.random() * trainDatas.length)
                batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
            }
            let test = trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => {
                const { z, q_z } = enc_fn(test)

                let out = <tf.Tensor4D>dec_fn(q_z)
                return out.squeeze([0])
            })
            let test_print = tf.tidy(() => tf.image.resizeNearestNeighbor(tf.concat([test_in, test_out], 1), [64, 128]))

            await tf.browser.toPixels(test_print, canvas)

            test.dispose()
            test_in.dispose()
            test_out.dispose()
            test_print.dispose()
        }
    }
    ;(<HTMLButtonElement>document.getElementById("train-ssim")).onclick = async () => {
        let fileIdx = Math.floor(Math.random() * trainDatas.length)
        let batchSize = 8
        let batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let batch = <tf.Tensor4D>trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    batch = tf.concat([batch, batch.reverse(2), batch.reverse(3), batch.reverse([2, 3])])

                    let z: tf.Tensor2D,
                        q_z: tf.Tensor2D = tf.tensor2d([[0]])
                    const grads1 = opt.computeGradients(
                        () => {
                            ;({ z, q_z } = enc_fn(batch))
                            q_z = tf.keep(q_z)
                            let out = <tf.Tensor4D>dec_fn(z)
                            let mainLoss = <tf.Scalar>tf.add(1, nn.ssim2d(batch, out, 11).mean().neg())
                            let l2Loss = [...enc_ws()]
                                .reduce((loss, w) => loss.add(w.square().mean()), tf.scalar(0))
                                .div([...enc_ws()].length)

                            return mainLoss.add(l2Loss)
                        },
                        <tf.Variable[]>(<unknown>[...enc_ws()])
                    ).grads
                    const grads2 = opt.computeGradients(
                        () => {
                            let out = <tf.Tensor4D>dec_fn(q_z)
                            let mainLoss = <tf.Scalar>tf.add(1, nn.ssim2d(batch, out, 11).mean().neg())
                            mainLoss.print()
                            let l2Loss = [...dec_ws()]
                                .reduce((loss, w) => loss.add(w.square().mean()), tf.scalar(0))
                                .div([...dec_ws()].length)

                            return mainLoss.add(l2Loss)
                        },
                        <tf.Variable[]>(<unknown>[...dec_ws()])
                    ).grads

                    opt.applyGradients(<[]>(<unknown>{ ...grads1, ...grads2 }))
                    q_z.dispose()
                })

                fileIdx = Math.floor(Math.random() * trainDatas.length)
                batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
            }
            let test = trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => {
                const { q_z } = enc_fn(test)

                let out = <tf.Tensor4D>dec_fn(q_z)
                return out.squeeze([0])
            })
            let test_print = tf.tidy(() => tf.image.resizeNearestNeighbor(tf.concat([test_in, test_out], 1), [64, 128]))

            await tf.browser.toPixels(test_print, canvas)

            test.dispose()
            test_in.dispose()
            test_out.dispose()
            test_print.dispose()
        }
    }
    ;(<HTMLButtonElement>document.getElementById("train-mix")).onclick = async () => {
        let fileIdx = Math.floor(Math.random() * trainDatas.length)
        let batchSize = 8
        let batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let batch = <tf.Tensor4D>trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    batch = tf.concat([batch, batch.reverse(2), batch.reverse(3), batch.reverse([2, 3])])

                    let z: tf.Tensor2D,
                        q_z: tf.Tensor2D = tf.tensor2d([[0]])
                    const grads1 = opt.computeGradients(
                        () => {
                            ;({ z, q_z } = enc_fn(batch))
                            q_z = tf.keep(q_z)
                            let out = <tf.Tensor4D>dec_fn(z)
                            let mainLoss = <tf.Scalar>(
                                tf
                                    .add(
                                        tf.losses.logLoss(batch, out).mul(2),
                                        nn.ssim2d(batch, out, 11).mean().log().neg()
                                    )
                                    .div(2)
                            )
                            let l2Loss = [...enc_ws()]
                                .reduce((loss, w) => loss.add(w.square().mean()), tf.scalar(0))
                                .div([...enc_ws()].length)

                            return mainLoss.add(l2Loss)
                        },
                        <tf.Variable[]>(<unknown>[...enc_ws()])
                    ).grads
                    const grads2 = opt.computeGradients(
                        () => {
                            let out = <tf.Tensor4D>dec_fn(q_z)
                            let mainLoss = <tf.Scalar>(
                                tf
                                    .add(
                                        tf.losses.logLoss(batch, out).mul(2),
                                        nn.ssim2d(batch, out, 11).mean().log().neg()
                                    )
                                    .div(2)
                            )
                            mainLoss.print()
                            let l2Loss = [...dec_ws()]
                                .reduce((loss, w) => loss.add(w.square().mean()), tf.scalar(0))
                                .div([...dec_ws()].length)

                            return mainLoss.add(l2Loss)
                        },
                        <tf.Variable[]>(<unknown>[...dec_ws()])
                    ).grads

                    opt.applyGradients(<[]>(<unknown>{ ...grads1, ...grads2 }))
                    q_z.dispose()
                })

                fileIdx = Math.floor(Math.random() * trainDatas.length)
                batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
            }
            let test = trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => {
                const { q_z } = enc_fn(test)

                let out = <tf.Tensor4D>dec_fn(q_z)
                return out.squeeze([0])
            })
            let test_print = tf.tidy(() => tf.image.resizeNearestNeighbor(tf.concat([test_in, test_out], 1), [64, 128]))

            await tf.browser.toPixels(test_print, canvas)

            test.dispose()
            test_in.dispose()
            test_out.dispose()
            test_print.dispose()
        }
    }
    ;(<HTMLButtonElement>document.getElementById("save-weights")).onclick = () => {
        tf.tidy(() => {
            let wList = [...enc_ws(), ...dec_ws()].reduce((acc, w) => {
                return [
                    ...acc,
                    { name: w.name, values: <Float32Array>w.dataSync() },
                    { name: `shape:${w.name}`, values: new Int32Array(w.shape) },
                ]
            }, [] as { name: string; values: Float32Array | Int32Array }[])

            let blob = myz.save(wList)
            let a = document.createElement("a")
            let url = window.URL.createObjectURL(blob)
            let filename = `weights.myz`
            a.href = url
            a.download = filename
            a.click()
            window.URL.revokeObjectURL(url)
        })
    }
    ;(<HTMLButtonElement>document.getElementById("load-weights")).onclick = () => {
        tf.tidy(() => {
            tf.tidy(() => {
                let load = document.createElement("input")
                load.type = "file"
                load.accept = ".myz"

                load.onchange = (event) => {
                    const files = <FileList>load.files
                    var reader = new FileReader()
                    reader.addEventListener("loadend", () => {
                        myz.load(<ArrayBuffer>reader.result).then((wList) => {
                            ;[...enc_ws(), ...dec_ws()].forEach((w) => {
                                let values = <Float32Array>wList.find((_w) => _w.name == w.name)?.values
                                let shape = <Int32Array>wList.find((_w) => _w.name == `shape:${w.name}`)?.values
                                if (values) {
                                    try {
                                        w.assign(tf.tensor(values, Array.from(shape)))
                                    } catch (e) {
                                        console.error(e)
                                    }
                                }
                            })
                        })
                    })
                    reader.readAsArrayBuffer(files[0])
                }

                load.click()
            })
        })
    }
    ;(<HTMLButtonElement>document.getElementById("test")).onclick = () => {
        tf.tidy(() => {
            let fileIdx = Math.floor(Math.random() * trainDatas.length)
            let batchSize = 1
            let batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize))
            let test = trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => {
                const { z, q_z } = enc_fn(test)
                let out = <tf.Tensor4D>dec_fn(q_z)
                return out.squeeze([0])
            })
            let test_print = tf.tidy(() => tf.image.resizeNearestNeighbor(tf.concat([test_in, test_out], 1), [64, 128]))

            tf.browser.toPixels(test_print, canvas)
        })
    }
    const testFPS = false // true
    if (testFPS) {
        const dk = 2,
            H = 8,
            W = 16
        const mha = MHA(H * W * dk, 8, 32, 32)
        const ff = FF(H * W * dk, H * W * dk * 2)
        const conv = tf.sequential({
            layers: [
                tf.layers.inputLayer({ inputShape: [64, 1, H * W * dk] }),
                tf.layers.separableConv2d({ filters: H * W * dk, kernelSize: [8, 1], padding: "valid" }),
            ],
        })
        const tt = () =>
            tf.tidy(() => {
                let test = tf.ones([1, 64, 1, H * W * dk])
                test = tf.pad(test, [
                    [0, 0],
                    [7, 0],
                    [0, 0],
                    [0, 0],
                ])
                test = (<tf.Tensor>conv.apply(test)).reshape([1, 64, H * W * dk])
                test = mha.fn(test, test)
                test = <tf.Tensor>ff.fn(test)
                test = mha.fn(test, test)
                test = <tf.Tensor>ff.fn(test)
                test = mha.fn(test, test).slice([0, 63, 0], [1, 1, -1])
                test = (<tf.Tensor>ff.fn(test)).reshape([1, H, W, dk])
                // let test = tf.ones([1, H, W, dk])
                tf.browser.toPixels(<tf.Tensor3D>dec_fn(test.reshape([1, H * W * dk])).squeeze([0]), canvas)
                requestAnimationFrame(tt)
            })
        tt()
    }
})
