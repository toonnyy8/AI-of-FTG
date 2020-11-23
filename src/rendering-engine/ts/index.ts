import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as myz from "./myz"
import { AED } from "./ae"
import { MHA, FF } from "./mha"

let canvas = <HTMLCanvasElement>document.getElementById("canvas")

tf.setBackend("webgl").then(() => {
    let [{ fn: enc_fn, ws: enc_ws }, { fn: dec_fn, ws: dec_ws }] = AED()
    let count = 0
    let trainData: tf.Tensor4D = tf.ones([1, 1, 1, 1])
    ;(<HTMLButtonElement>document.getElementById("load-data")).onclick = () => {
        tf.tidy(() => {
            let load = document.createElement("input")
            load.type = "file"
            load.accept = ".myz"

            load.onchange = (event) => {
                const files = <FileList>load.files
                var reader = new FileReader()
                reader.addEventListener("loadend", () => {
                    myz.load(<ArrayBuffer>reader.result).then((datas) => {
                        let frames = datas.find((data) => {
                            return data.name == "frames"
                        })
                        let frameShape = datas.find((data) => {
                            return data.name == "frameShape"
                        })
                        trainData.dispose()
                        trainData = tf.keep(
                            tf
                                .tensor4d(
                                    <Uint8Array>frames?.values,
                                    <[number, number, number, number]>Array.from(<Uint8Array>frameShape?.values)
                                )
                                .cast("float32")
                                .div(255)
                        )
                    })
                })
                reader.readAsArrayBuffer(files[0])
            }

            load.click()
        })
    }
    const op = tf.train.adamax(0.01)
    ;(<HTMLButtonElement>document.getElementById("train-log")).onclick = async () => {
        let batchSize = 8
        let batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let batch = <tf.Tensor4D>trainData.slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    batch = tf.concat([batch, batch.reverse(2), batch.reverse(3), batch.reverse([2])])
                    op.minimize(
                        () => {
                            let out = <tf.Tensor4D>dec_fn(enc_fn(batch))
                            return tf.losses.logLoss(batch, out)
                        },
                        true,
                        <tf.Variable[]>(<unknown>[...enc_ws(), ...dec_ws()])
                    )?.print()
                })

                batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
            }
            let test = trainData.slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => (<tf.Tensor>dec_fn(enc_fn(test))).squeeze([0]))
            let test_print = tf.concat([test_in, test_out], 1)

            await tf.browser.toPixels(test_print, canvas)

            test.dispose()
            test_in.dispose()
            test_out.dispose()
            test_print.dispose()
        }
    }
    ;(<HTMLButtonElement>document.getElementById("train-ssim")).onclick = async () => {
        let batchSize = 8
        let batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let batch = <tf.Tensor4D>trainData.slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    batch = tf.concat([batch, batch.reverse(2), batch.reverse(3), batch.reverse([2])])
                    op.minimize(
                        () => {
                            let out = <tf.Tensor4D>dec_fn(enc_fn(batch))
                            return <tf.Scalar>tf.add(1, nn.ssim2d(batch, out, 5).mean().neg())
                        },
                        true,
                        <tf.Variable[]>(<unknown>[...enc_ws(), ...dec_ws()])
                    )?.print()
                })

                batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
            }
            let test = trainData.slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => (<tf.Tensor>dec_fn(enc_fn(test))).squeeze([0]))
            let test_print = tf.concat([test_in, test_out], 1)

            await tf.browser.toPixels(test_print, canvas)

            test.dispose()
            test_in.dispose()
            test_out.dispose()
            test_print.dispose()
        }
    }
    ;(<HTMLButtonElement>document.getElementById("train-mix")).onclick = async () => {
        let batchSize = 8
        let batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let batch = <tf.Tensor4D>trainData.slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    batch = tf.concat([batch, batch.reverse(2), batch.reverse(3), batch.reverse([2])])
                    op.minimize(
                        () => {
                            let out = <tf.Tensor4D>dec_fn(enc_fn(batch))
                            return <tf.Scalar>(
                                tf
                                    .add(
                                        tf.losses.logLoss(batch, out).mul(2),
                                        nn.ssim2d(batch, out, 5).mean().log().neg()
                                    )
                                    .div(3)
                            )
                        },
                        true,
                        <tf.Variable[]>(<unknown>[...enc_ws(), ...dec_ws()])
                    )?.print()
                })

                batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
            }
            let test = trainData.slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => (<tf.Tensor>dec_fn(enc_fn(test))).squeeze([0]))
            let test_print = tf.concat([test_in, test_out], 1)

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
            let batchSize = 1
            let batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
            let test = trainData.slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
            let test_in = <tf.Tensor3D>test.squeeze([0])
            let test_out = <tf.Tensor3D>tf.tidy(() => (<tf.Tensor>dec_fn(enc_fn(test))).squeeze([0]))
            let test_print = tf.concat([test_in, test_out], 1)

            tf.browser.toPixels(test_print, canvas)
        })
    }
    // let mha = MHA(100, 8, 32, 32)
    // let ff = FF(100, 256)
    // let conv = tf.sequential({
    //     layers: [
    //         tf.layers.inputLayer({ inputShape: [64, 1, 100] }),
    //         tf.layers.separableConv2d({ filters: 100, kernelSize: [8, 1], padding: "valid" }),
    //     ],
    // })
    // let tt = () =>
    //     tf.tidy(() => {
    //         let test = tf.ones([1, 64, 1, 100])
    //         test = tf.pad(test, [
    //             [0, 0],
    //             [7, 0],
    //             [0, 0],
    //             [0, 0],
    //         ])
    //         test = (<tf.Tensor>conv.apply(test)).reshape([1, 64, 100])
    //         test = mha.fn(test, test)
    //         test = <tf.Tensor>ff.fn(test)
    //         test = mha.fn(test, test).slice([0, 63, 0], [1, 1, -1])
    //         test = (<tf.Tensor>ff.fn(test)).reshape([1, 5, 5, 4])
    //         // let test = tf.ones([1, 5, 5, 4])
    //         tf.browser.toPixels(<tf.Tensor3D>dec_fn(test).squeeze([0]), canvas)
    //         requestAnimationFrame(tt)
    //     })
    // tt()
})
