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
    let [{ fn: enc_fn, ws: enc_ws }, { fn: dec_fn, ws: dec_ws }, { mapping }] = AED({
        assetGroups: 8,
        assetSize: 16,
        assetNum: 32,
        dk: dk,
    })
    let driver = Driver({
        ctrlNum: 2,
        actionNum: 2 ** 7,
        dact: 64,
        dinp: h * w * dk,
        dmodel: 256,
        r: 8,
        head: 8,
        dk: 32,
        dv: 32,
        hiddens: 512,
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
                let test_yImgs = <tf.Tensor3D[]>(
                    dec_fn(
                        enc_fn(
                            trainDatas[fileIdx]
                                .slice([batchStart, 0, 0, 0], [batchSize, -1, -1, -1])
                                .slice([batchSize - 2, 0, 0, 0], [2, -1, -1, -1])
                        )
                    ).unstack(0)
                )
                let ctrl1Batch = ctrl1s[fileIdx].slice(batchStart, batchStart + batchSize)
                let ctrl2Batch = ctrl2s[fileIdx].slice(batchStart, batchStart + batchSize)
                let test_out = <tf.Tensor3D>(
                    tf.tidy(() =>
                        (<tf.Tensor>dec_fn(driver.fn(<tf.Tensor4D>enc_fn(test_xImg), [ctrl1Batch, ctrl2Batch])))
                            .slice([batchSize - 1, 0, 0, 0], [1, -1, -1, -1])
                            .squeeze([0])
                    )
                )
                let test_print = tf.concat([...test_yImgs, test_out], 1)

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
                    let lossWeight = tf.linspace(0, 1, 64) //.sqrt()
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
                        return ctrl1s[fileIdx].slice(batchStart + offset, batchStart + batchSize + offset)
                    })
                    let x_enc = <tf.Tensor4D>enc_fn(xImg)
                    let y_encs = yImgs.map((yImg) => <tf.Tensor4D>enc_fn(yImg))
                    let mask = (() => {
                        let arr = new Array(batchSize).fill(0)
                        arr[arr.length - 1] = 1
                        return tf.tensor4d(arr, [batchSize, 1, 1, 1])
                    })()
                    // let mask = lossWeight.reshape([-1, 1, 1, 1])
                    opt.minimize(
                        () => {
                            const loss_fn = <T extends tf.Tensor>(y: T, _y: T, axis: number[]) => {
                                return <tf.Scalar>tf.sub(y, _y).square().mean(axis).mul(lossWeight).sum()
                            }
                            const mainLoss = y_encs
                                .reduce(
                                    (prev, y_enc, idx) => {
                                        let next_enc = <tf.Tensor4D>(
                                            driver.fn(prev.enc, [ctrl1Batchs[idx], ctrl2Batchs[idx]])
                                        )

                                        return {
                                            loss: <tf.Scalar>prev.loss.add(loss_fn(y_enc, next_enc, [1, 2, 3])),
                                            enc: <tf.Tensor4D>(
                                                tf.concat([
                                                    prev.enc.slice([1, 0, 0, 0], [-1, -1, -1, -1]),
                                                    next_enc.slice([batchSize - 1, 0, 0, 0], [1, -1, -1, -1]),
                                                ])
                                            ), //next_enc, //tf.add(next_enc.mul(mask), y_enc.mul(tf.sub(1, mask)))
                                        }
                                    },
                                    { loss: tf.scalar(0), enc: x_enc }
                                )
                                .loss.div(t)
                            mainLoss.print()
                            const l2Loss = driver
                                .ws()
                                .reduce((loss, ws) => loss.add(ws.square().mean()), tf.scalar(0))
                                .div(driver.ws().length)
                            return mainLoss.add(l2Loss)
                        },
                        false,
                        <tf.Variable[]>(<unknown>[...driver.ws()])
                    )?.print()
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
    let p1Ctrl = new Array(7).fill(false)
    let p2Ctrl = new Array(7).fill(false)
    document.addEventListener("keydown", (event) => {
        switch (event.code) {
            case "KeyW":
                p1Ctrl[0] = true
                break
            case "KeyS":
                p1Ctrl[1] = true
                break
            case "KeyA":
                p1Ctrl[2] = true
                break
            case "KeyD":
                p1Ctrl[3] = true
                break
            case "KeyJ":
                p1Ctrl[4] = true
                break
            case "KeyK":
                p1Ctrl[5] = true
                break
            case "KeyL":
                p1Ctrl[6] = true
                break

            case "ArrowUp":
                p2Ctrl[0] = true
                break
            case "ArrowDown":
                p2Ctrl[1] = true
                break
            case "ArrowLeft":
                p2Ctrl[2] = true
                break
            case "ArrowRight":
                p2Ctrl[3] = true
                break
            case "Numpad1":
                p2Ctrl[4] = true
                break
            case "Numpad2":
                p2Ctrl[5] = true
                break
            case "Numpad3":
                p2Ctrl[6] = true
                break
        }
    })

    document.addEventListener("keyup", (event) => {
        switch (event.code) {
            case "KeyW":
                p1Ctrl[0] = false
                break
            case "KeyS":
                p1Ctrl[1] = false
                break
            case "KeyA":
                p1Ctrl[2] = false
                break
            case "KeyD":
                p1Ctrl[3] = false
                break
            case "KeyJ":
                p1Ctrl[4] = false
                break
            case "KeyK":
                p1Ctrl[5] = false
                break
            case "KeyL":
                p1Ctrl[6] = false
                break

            case "ArrowUp":
                p2Ctrl[0] = false
                break
            case "ArrowDown":
                p2Ctrl[1] = false
                break
            case "ArrowLeft":
                p2Ctrl[2] = false
                break
            case "ArrowRight":
                p2Ctrl[3] = false
                break
            case "Numpad1":
                p2Ctrl[4] = false
                break
            case "Numpad2":
                p2Ctrl[5] = false
                break
            case "Numpad3":
                p2Ctrl[6] = false
                break
        }
    })
    let runTest = false
    ;(<HTMLButtonElement>document.getElementById("test")).onclick = () => {
        if (runTest) runTest = false
        else {
            runTest = true

            const L = 64
            let fileIdx = Math.floor(Math.random() * trainDatas.length)

            let input_enc = <tf.Tensor4D>enc_fn(trainDatas[fileIdx].slice([0, 0, 0, 0], [L, -1, -1, -1]))
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
                    let next_enc = <tf.Tensor4D>driver.fn(input_enc, [input_ctrl1, input_ctrl2])
                    next_enc = tf.concat([
                        input_enc.slice([1, 0, 0, 0], [-1, -1, -1, -1]),
                        next_enc.slice([L - 1, 0, 0, 0], [1, -1, -1, -1]),
                    ])
                    input_enc.dispose()
                    input_enc = tf.keep(next_enc)
                    input_ctrl1 = [...input_ctrl1.slice(1), p1Ctrl.reduce((prev, curr) => prev * 2 + Number(curr), 0)]
                    input_ctrl2 = [...input_ctrl2.slice(1), p2Ctrl.reduce((prev, curr) => prev * 2 + Number(curr), 0)]

                    tf.browser.toPixels(
                        <tf.Tensor3D>dec_fn(next_enc.slice([L - 1, 0, 0, 0], [1, -1, -1, -1])).squeeze([0]),
                        canvas
                    )
                })
                if (runTest) requestAnimationFrame(tt)
                else input_enc.dispose()
            }
            tt()
        }
    }
})
