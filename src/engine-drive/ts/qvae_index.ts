import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as myz from "./myz"
import { QVAE } from "./qvae"
import { Driver } from "./qvae_driver"

let canvas = <HTMLCanvasElement>document.getElementById("canvas")

tf.setBackend("webgl").then(() => {
    let {
        enc: { fn: enc_fn, ws: enc_ws },
        dec: { fn: dec_fn, synthesizer, ws: dec_ws },
    } = QVAE({ hiddens: 512 })
    let driver = Driver({
        ctrlNum: 2,
        actionNum: 36,
        dinp: 512,
        dmodel: 128,
        head: 8,
        dk: 32,
        dv: 32,
        hiddens: 1024,
        restrictHead: 32,
        layerNum: 2,
    })

    let trainDatas: tf.Tensor4D[] = []
    let ctrl1s: number[][] = []
    let ctrl2s: number[][] = []

    const test = (batchSize: number) => {
        return new Promise((resolve, reject) => {
            let fileIdx = Math.floor(Math.random() * trainDatas.length)
            let batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize - 1))
            tf.tidy(() => {
                let test_xImg = trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                let test_yImgs = <tf.Tensor3D[]>tf
                    .tidy(() => {
                        const { q_z } = enc_fn(
                            trainDatas[fileIdx]
                                .slice([batchStart + 1, 0, 0, 0], [batchSize, -1, -1, -1])
                                .slice([batchSize - 2, 0, 0, 0], [2, -1, -1, -1])
                        )
                        let out = <tf.Tensor4D>dec_fn(q_z)
                        return out
                    })
                    .unstack(0)
                let ctrl1Batch = ctrl1s[fileIdx].slice(batchStart, batchStart + batchSize)
                let ctrl2Batch = ctrl2s[fileIdx].slice(batchStart, batchStart + batchSize)

                let test_out = <tf.Tensor3D>tf.tidy(() =>
                    (<tf.Tensor>dec_fn(
                        driver
                            .fn(
                                tf.tidy(() => {
                                    const { q_z } = enc_fn(test_xImg)
                                    return q_z
                                }),
                                [ctrl1Batch, ctrl2Batch]
                            )
                            .round()
                    ))
                        .slice([batchSize - 1, 0, 0, 0], [1, -1, -1, -1])
                        .squeeze([0])
                )
                // enc_fn(
                //     trainDatas[fileIdx]
                //         .slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                //         .slice([batchSize - 1, 0, 0, 0], [1, -1, -1, -1])
                // )
                //     .squeeze([0])
                //     .sub(
                //         driver
                //             .fn(<tf.Tensor2D>enc_fn(test_xImg), [ctrl1Batch, ctrl2Batch])
                //             .slice([batchSize - 1, 0], [1, -1])
                //             .squeeze([0])
                //     )
                //     .array()
                //     .then((arr) => console.log(arr))
                let test_print = tf.image.resizeNearestNeighbor(tf.concat([...test_yImgs, test_out], 1), [64, 64 * 3])

                resolve(tf.browser.toPixels(test_print, canvas))
            })
        })
    }

    ;(<HTMLButtonElement>document.getElementById("load-data")).onclick = () => {
        let load = document.createElement("input")
        load.type = "file"
        load.accept = ".myz"
        load.multiple = true

        load.onchange = (event) => {
            const files = Array.from(<FileList>load.files)
            trainDatas.forEach((trainData) => trainData.dispose())
            trainDatas = []
            ctrl1s = []
            ctrl2s = []

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
                            ctrl1s = [
                                ...ctrl1s,
                                Array.from(
                                    <Uint8Array>datas.find((data) => {
                                        return data.name == "ctrl1"
                                    })?.values
                                ),
                            ]
                            ctrl2s = [
                                ...ctrl2s,
                                Array.from(
                                    <Uint8Array>datas.find((data) => {
                                        return data.name == "ctrl2"
                                    })?.values
                                ),
                            ]
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

    const opt = tf.train.adamax(0.001)
    ;(<HTMLButtonElement>document.getElementById("train")).onclick = async () => {
        let batchSize = 64
        let t = 1
        const train = (loop: number = 64) => {
            for (let j = 0; j < loop; j++) {
                let fileIdx = Math.floor(Math.random() * trainDatas.length)
                let batchStart = Math.round(Math.random() * (trainDatas[fileIdx].shape[0] - batchSize - t))
                tf.tidy(() => {
                    let lossWeight = <tf.Tensor2D>tf.linspace(0, 1, 64).reshape([64, 1]) //.sqrt()
                    lossWeight = lossWeight.div(lossWeight.sum())
                    let xImg = <tf.Tensor4D>trainDatas[fileIdx].slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                    let yImgs = new Array(t).fill(0).map((_, offset) => {
                        return <tf.Tensor4D>(
                            trainDatas[fileIdx].slice([batchStart + offset + 1, 0, 0, 0], [batchSize, -1, -1, -1])
                        )
                    })
                    let ctrl1Batchs = new Array(t).fill(0).map((_, offset) => {
                        return ctrl1s[fileIdx].slice(batchStart + offset, batchStart + batchSize + offset)
                    })
                    let ctrl2Batchs = new Array(t).fill(0).map((_, offset) => {
                        return ctrl2s[fileIdx].slice(batchStart + offset, batchStart + batchSize + offset)
                    })
                    let x_enc = tf.tidy(() => {
                        const { q_z } = enc_fn(xImg)
                        let rate = Math.random() * 0.5
                        let mask = tf.randomUniform(q_z.shape, 0, 1).greater(rate).cast("float32")
                        let randQ = tf.randomUniform(q_z.shape, -1.49, 1.49).round()
                        return <tf.Tensor2D>tf.mul(mask, q_z).add(tf.mul(tf.sub(1, mask), randQ))

                        // return q_z
                    })
                    let y_encs = yImgs.map((yImg) =>
                        tf.tidy(() => {
                            const { q_z } = enc_fn(yImg)
                            return q_z
                        })
                    )
                    let mask = (() => {
                        let arr = new Array(batchSize).fill(0)
                        arr[arr.length - 1] = 1
                        return tf.tensor2d(arr, [batchSize, 1])
                    })()
                    // let mask = lossWeight.reshape([-1, 1, 1, 1])
                    const logLoss = (label: tf.Tensor2D, pred: tf.Tensor2D): tf.Scalar => {
                        let eps = 0.0000001
                        let p = tf.clipByValue(pred, eps, 1 - eps)
                        return tf.mean(
                            label
                                .mul(tf.log(p))
                                .neg()
                                .sub(tf.sub(1, label).mul(tf.log(tf.sub(1, p))))
                                .mul(lossWeight)
                                .sum(0)
                        )
                    }
                    const loss_fn = <T extends tf.Tensor2D>(y: T, _y: T): tf.Scalar => {
                        return tf.sub(y, _y).square().mean(-1, true).mul(lossWeight).sum()
                        // return logLoss(y.mul(0.5).add(0.5), _y.mul(0.5).add(0.5))
                    }
                    const { gradients, loss } = y_encs.reduce(
                        (prev, y_enc, idx) => {
                            let next_enc: tf.Tensor2D
                            let { value, grads } = opt.computeGradients(
                                () =>
                                    tf.tidy(() => {
                                        next_enc = tf.keep(
                                            <tf.Tensor2D>driver.fn(prev.enc, [ctrl1Batchs[idx], ctrl2Batchs[idx]])
                                        )
                                        return loss_fn(y_enc, next_enc)
                                    }),
                                <tf.Variable[]>(<unknown>[...driver.ws()])
                            )
                            let enc = tf.tidy(
                                () =>
                                    <tf.Tensor2D>(
                                        tf.concat([
                                            prev.enc.slice([1, 0], [-1, -1]),
                                            next_enc.slice([batchSize - 1, 0], [1, -1]).round(),
                                        ])
                                    )
                            )
                            // @ts-ignore
                            next_enc.dispose()
                            prev.enc.dispose()
                            let gradients = tf.tidy(() =>
                                Object.keys(grads).reduce((gradients, name) => {
                                    let g = grads[name].where(grads[name].isFinite(), 0)
                                    if (prev.gradients[name] !== undefined) {
                                        gradients[name] = prev.gradients[name].add(g)
                                        prev.gradients[name].dispose()
                                    } else gradients[name] = g.clone()

                                    grads[name].dispose()
                                    g.dispose()
                                    return gradients
                                }, <tf.NamedTensorMap>{})
                            )

                            return {
                                gradients: gradients,
                                enc: enc,
                                loss: <tf.Scalar>prev.loss.add(value),
                            }
                        },
                        { gradients: <tf.NamedTensorMap>{}, enc: x_enc.clone(), loss: tf.scalar(0) }
                    )
                    loss.div(t).print()
                    Object.keys(gradients).map((name) => {
                        gradients[name] = gradients[name].div(t)
                    })
                    opt.applyGradients(<[]>(<unknown>gradients))
                })
            }
        }

        const trainLoop = (epochs: number) => {
            test(batchSize).then(() => {
                train(64)
                if (epochs > 0) {
                    trainLoop(epochs - 1)
                }
            })
        }
        trainLoop(64)
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
    let arrow1 = 5
    let attack1 = 0
    let arrow2 = 5
    let attack2 = 0
    document.addEventListener("keydown", (event) => {
        switch (event.code) {
            case "KeyZ":
                arrow1 = 1
                break
            case "KeyX":
                arrow1 = 2
                break
            case "KeyC":
                arrow1 = 3
                break
            case "KeyA":
                arrow1 = 4
                break
            case "KeyD":
                arrow1 = 6
                break
            case "KeyQ":
                arrow1 = 7
                break
            case "KeyW":
                arrow1 = 8
                break
            case "KeyE":
                arrow1 = 9
                break
            case "KeyJ":
                attack1 = 1
                break
            case "KeyK":
                attack1 = 2
                break
            case "KeyL":
                attack1 = 3
                break

            case "Digit1":
            case "Numpad1":
                arrow2 = 1
                break
            case "Digit2":
            case "Numpad2":
                arrow2 = 2
                break
            case "Digit3":
            case "Numpad3":
                arrow2 = 3
                break
            case "Digit4":
            case "Numpad4":
                arrow2 = 4
                break
            case "Digit6":
            case "Numpad6":
                arrow2 = 6
                break
            case "Digit7":
            case "Numpad7":
                arrow2 = 7
                break
            case "Digit8":
            case "Numpad8":
                arrow2 = 8
                break
            case "Digit9":
            case "Numpad9":
                arrow2 = 9
                break
            case "Digit0":
            case "Numpad0":
                attack2 = 1
                break
            case "Minus":
            case "NumpadSubtract":
                attack2 = 2
                break
            case "Equal":
            case "NumpadAdd":
                attack2 = 3
                break
        }
    })

    document.addEventListener("keyup", (event) => {
        switch (event.code) {
            case "KeyZ":
                if (arrow1 == 1) arrow1 = 5
                break
            case "KeyX":
                if (arrow1 == 2) arrow1 = 5
                break
            case "KeyC":
                if (arrow1 == 3) arrow1 = 5
                break
            case "KeyA":
                if (arrow1 == 4) arrow1 = 5
                break
            case "KeyD":
                if (arrow1 == 6) arrow1 = 5
                break
            case "KeyQ":
                if (arrow1 == 7) arrow1 = 5
                break
            case "KeyW":
                if (arrow1 == 8) arrow1 = 5
                break
            case "KeyE":
                if (arrow1 == 9) arrow1 = 5
                break
            case "KeyJ":
                if (attack1 == 1) attack1 = 0
                break
            case "KeyK":
                if (attack1 == 2) attack1 = 0
                break
            case "KeyL":
                if (attack1 == 3) attack1 = 0
                break

            case "Digit1":
            case "Numpad1":
                if (arrow2 == 1) arrow2 = 5
                break
            case "Digit2":
            case "Numpad2":
                if (arrow2 == 2) arrow2 = 5
                break
            case "Digit3":
            case "Numpad3":
                if (arrow2 == 3) arrow2 = 5
                break
            case "Digit4":
            case "Numpad4":
                if (arrow2 == 4) arrow2 = 5
                break
            case "Digit6":
            case "Numpad6":
                if (arrow2 == 6) arrow2 = 5
                break
            case "Digit7":
            case "Numpad7":
                if (arrow2 == 7) arrow2 = 5
                break
            case "Digit8":
            case "Numpad8":
                if (arrow2 == 8) arrow2 = 5
                break
            case "Digit9":
            case "Numpad9":
                if (arrow2 == 9) arrow2 = 5
                break
            case "Digit0":
            case "Numpad0":
                if (attack2 == 1) attack2 = 0
                break
            case "Minus":
            case "NumpadSubtract":
                if (attack2 == 2) attack2 = 0
                break
            case "Equal":
            case "NumpadAdd":
                if (attack2 == 3) attack2 = 0
                break
        }
    })
    let runTest = false
    ;(<HTMLButtonElement>document.getElementById("test")).onclick = () => {
        // test(64)
        if (runTest) runTest = false
        else {
            runTest = true

            const L = 64
            let fileIdx = Math.floor(Math.random() * trainDatas.length)
            let input_enc = tf.tidy(() => {
                const { q_z } = enc_fn(trainDatas[fileIdx].slice([0, 0], [L, -1]))
                return q_z
            })
            let input_ctrl1 = ctrl1s[fileIdx].slice(0, L)
            let input_ctrl2 = ctrl2s[fileIdx].slice(0, L)

            const tt = () => {
                tf.tidy(() => {
                    // const p1Ctrl = [
                    //     Math.random() < 0.3 ? true : false,
                    //     Math.random() < 0.1 ? true : false,
                    //     Math.random() < 0.5 ? true : false,
                    //     Math.random() < 0.5 ? true : false,
                    //     Math.random() < 0.01 ? true : false,
                    //     Math.random() < 0.01 ? true : false,
                    //     Math.random() < 0.01 ? true : false,
                    // ]

                    // const p2Ctrl = [
                    //     Math.random() < 0.2 ? true : false,
                    //     Math.random() < 0.2 ? true : false,
                    //     Math.random() < 0.4 ? true : false,
                    //     Math.random() < 0.4 ? true : false,
                    //     Math.random() < 0.1 ? true : false,
                    //     Math.random() < 0.1 ? true : false,
                    //     Math.random() < 0.1 ? true : false,
                    // ]
                    let next_enc = <tf.Tensor2D>driver.fn(input_enc, [input_ctrl1, input_ctrl2])
                    next_enc = tf.concat([
                        input_enc.slice([1, 0], [-1, -1]),
                        next_enc.slice([L - 1, 0], [1, -1]).round(),
                    ])
                    input_enc.dispose()
                    input_enc = tf.keep(next_enc)
                    input_ctrl1 = [...input_ctrl1.slice(1), (arrow1 - 1) * 4 + attack1]
                    input_ctrl2 = [...input_ctrl2.slice(1), (arrow2 - 1) * 4 + attack2]

                    tf.browser.toPixels(<tf.Tensor3D>dec_fn(next_enc.slice([L - 1, 0], [1, -1])).squeeze([0]), canvas)
                })
                if (runTest) requestAnimationFrame(tt)
                else input_enc.dispose()
            }
            tt()
        }
    }
})
