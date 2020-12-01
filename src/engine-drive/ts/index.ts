import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as myz from "./myz"
import { AED } from "./ae"
import { MHA, FF } from "./mha"
import { Driver } from "./driver"
import { tensor, unstack } from "@tensorflow/tfjs"

let canvas = <HTMLCanvasElement>document.getElementById("canvas")

tf.setBackend("webgl").then(() => {
    let h = 5,
        w = 10,
        dk = 4
    let [{ fn: enc_fn, ws: enc_ws }, { fn: dec_fn, ws: dec_ws }, { att }] = AED({
        assetGroups: 8,
        assetSize: 16,
        assetNum: 32,
        dk: dk,
    })
    let driver = Driver({
        ctrlNum: 2,
        actionNum: 2 ** 7,
        dact: 64,
        dmodel: h * w * dk,
        r: 8,
        head: 8,
        dk: 32,
        dv: 32,
    })
    let trainData: tf.Tensor4D = tf.ones([1, 1, 1, 1])
    let ctrl1: number[]
    let ctrl2: number[]
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
                        ctrl1 = Array.from(
                            <Uint8Array>datas.find((data) => {
                                return data.name == "ctrl1"
                            })?.values
                        )
                        ctrl2 = Array.from(
                            <Uint8Array>datas.find((data) => {
                                return data.name == "ctrl2"
                            })?.values
                        )
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
    const op = tf.train.adamax(0.001)
    ;(<HTMLButtonElement>document.getElementById("train")).onclick = async () => {
        let batchSize = 64
        let t = 3
        let batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize - t))
        let train = () => {
            for (let j = 0; j < 64; j++) {
                tf.tidy(() => {
                    let lossWeight = tf.linspace(0, 1, 64) //.sqrt()
                    lossWeight = lossWeight.div(lossWeight.sum())
                    let xImg = <tf.Tensor4D>trainData.slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    let yImgs = new Array(t).fill(0).map((_, offset) => {
                        return <tf.Tensor4D>trainData.slice([batchStart + offset + 1, 0, 0, 0], [batchSize, -1, -1, -1])
                    })
                    let ctrl1Batchs = new Array(t).fill(0).map((_, offset) => {
                        return ctrl1.slice(batchStart + offset, batchStart + batchSize + offset)
                    })
                    let ctrl2Batchs = new Array(t).fill(0).map((_, offset) => {
                        return ctrl2.slice(batchStart + offset, batchStart + batchSize + offset)
                    })
                    let x_enc = <tf.Tensor4D>enc_fn(xImg)
                    let y_encs = yImgs.map((yImg) => <tf.Tensor4D>enc_fn(yImg))
                    let mask = (() => {
                        let arr = new Array(batchSize).fill(0)
                        arr[arr.length - 1] = 1
                        return tf.tensor4d(arr, [batchSize, 1, 1, 1])
                    })()

                    op.minimize(
                        () => {
                            const loss_fn = (y: tf.Tensor4D, _y: tf.Tensor4D) => {
                                return <tf.Scalar>tf.sub(y, _y).square().mean([1, 2, 3]).mul(lossWeight).sum()
                            }
                            return y_encs
                                .reduce(
                                    (prev, y_enc, idx) => {
                                        let next_enc = <tf.Tensor4D>(
                                            driver.fn(prev.enc, [ctrl1Batchs[idx], ctrl2Batchs[idx]])
                                        )
                                        return {
                                            loss: <tf.Scalar>prev.loss.add(loss_fn(y_enc, next_enc)),
                                            enc: <tf.Tensor4D>next_enc.mul(mask).add(y_enc.mul(tf.sub(1, mask))),
                                        }
                                    },
                                    { loss: tf.scalar(0), enc: x_enc }
                                )
                                .loss.div(t)
                        },
                        true,
                        <tf.Variable[]>(<unknown>[...driver.ws()])
                    )?.print()
                })

                batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize - t))
            }
            tf.tidy(() => {
                let test_xImg = trainData.slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                let test_yImgs = <tf.Tensor3D[]>(
                    dec_fn(
                        enc_fn(
                            trainData
                                .slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                                .slice([batchSize - 2, 0, 0, 0], [2, -1, -1, -1])
                        )
                    ).unstack(0)
                )
                // let test_yImg = <tf.Tensor3D>(
                //     dec_fn(
                //         enc_fn(trainData.slice([batchStart + batchSize + (t - 1), 0, 0, 0], [1, -1, -1, -1]))
                //     ).squeeze([0])
                // )
                let ctrl1Batch = ctrl1.slice(batchStart, batchStart + batchSize)
                let ctrl2Batch = ctrl2.slice(batchStart, batchStart + batchSize)
                let test_out = <tf.Tensor3D>(
                    tf.tidy(() =>
                        (<tf.Tensor>dec_fn(driver.fn(<tf.Tensor4D>enc_fn(test_xImg), [ctrl1Batch, ctrl2Batch])))
                            .slice([batchSize - 1, 0, 0, 0], [1, -1, -1, -1])
                            .squeeze([0])
                    )
                )
                let test_print = tf.concat([...test_yImgs, test_out], 1)

                tf.browser.toPixels(test_print, canvas)
            })
            return tf.nextFrame()
        }

        for (let i = 0; i < 64; i++) {
            await train()
        }
    }
    ;(<HTMLButtonElement>document.getElementById("save-weights")).onclick = () => {
        tf.tidy(() => {
            let wList = [...driver.ws()].reduce((acc, w) => {
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
                            ;[...driver.ws()].forEach((w) => {
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
    ;(<HTMLButtonElement>document.getElementById("load-rendering-weights")).onclick = () => {
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
    // ; (<HTMLButtonElement>document.getElementById("test")).onclick = () => {
    //     tf.tidy(() => {
    //         let batchSize = 1
    //         let batchStart = Math.round(Math.random() * (trainData.shape[0] - batchSize))
    //         let test = trainData.slice([batchStart, 0, 0, 0], [1, -1, -1, -1])
    //         let test_in = <tf.Tensor3D>test.squeeze([0])
    //         let test_out = <tf.Tensor3D>tf.tidy(() => (<tf.Tensor>dec_fn(enc_fn(test))).squeeze([0]))
    //         let test_print = tf.concat([test_in, test_out], 1)

    //         tf.browser.toPixels(test_print, canvas)
    //     })
    // }
    const testFPS = false
    if (testFPS) {
        const H = 5,
            W = 10
        const tt = () =>
            tf.tidy(() => {
                let test = <tf.Tensor4D>tf.ones([64, H, W, dk])
                test = <tf.Tensor4D>(
                    driver.fn(test, new Array(2).fill(new Array(64).fill(0))).slice([63, 0, 0, 0], [1, -1, -1, -1])
                )
                tf.browser.toPixels(<tf.Tensor3D>dec_fn(test).squeeze([0]), canvas)
                requestAnimationFrame(tt)
            })
        tt()
    }
})
