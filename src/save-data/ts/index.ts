import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as myz from "./myz"

interface KeySet {
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

const keySets: [KeySet, KeySet] = [
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

const control = (
    ctrl: {
        arrow: {
            jump: 1 | 0 | -1
            squat: 1 | 0 | -1
            left: 1 | 0 | -1
            right: 1 | 0 | -1
        }
        attack: {
            light: 1 | 0 | -1
            medium: 1 | 0 | -1
            heavy: 1 | 0 | -1
        }
    },
    keySet: KeySet
) => {
    Object.keys(ctrl.arrow).forEach((arrow) => {
        switch (ctrl.arrow[<"jump" | "squat" | "left" | "right">arrow]) {
            case 1:
                document.dispatchEvent(
                    new KeyboardEvent("keydown", {
                        key: keySet[<"jump" | "squat" | "left" | "right">arrow],
                    })
                )
                break
            case 0:
                break
            case -1:
                document.dispatchEvent(
                    new KeyboardEvent("keyup", {
                        key: keySet[<"jump" | "squat" | "left" | "right">arrow],
                    })
                )
                break
        }
    })

    Object.keys(ctrl.attack).forEach((attack) => {
        switch (ctrl.attack[<"light" | "medium" | "heavy">attack]) {
            case 1:
                document.dispatchEvent(
                    new KeyboardEvent("keydown", {
                        key: keySet.attack[<"light" | "medium" | "heavy">attack],
                    })
                )
                break
            case 0:
                break
            case -1:
                document.dispatchEvent(
                    new KeyboardEvent("keyup", {
                        key: keySet.attack[<"light" | "medium" | "heavy">attack],
                    })
                )
                break
        }
    })
}

const memory = (size: number) => {
    let mem: tf.Tensor[] = []
    const read = (idx: number) => {
        return tf.tidy(() => mem[idx])
    }
    const clone = () => {
        return mem.map((m) => m.clone())
    }
    const write = (...ms: tf.Tensor[]) => {
        ms.forEach((m) => {
            mem.push(m)
            if (mem.length > size) {
                mem.shift()?.dispose()
            }
        })
    }
    const clear = () => {
        mem.forEach((m) => m.dispose())
        mem = []
    }
    return {
        read,
        clone,
        write,
        clear,
    }
}

tf.setBackend("webgl")
    .then(() => Game(keySets, canvas))
    .then(({ next, render, getP1, getP2, getRestart }) => {
        let count = 0
        let { clone, write, clear } = memory(60 * 60)
        let { clone: ctrl1Clone, write: ctrl1Write, clear: ctrl1Clear } = memory(60 * 60)
        let { clone: ctrl2Clone, write: ctrl2Write, clear: ctrl2Clear } = memory(60 * 60)
        let blobs = [] as Blob[]

        ;(<HTMLButtonElement>document.getElementById("save")).onclick = () => {
            tf.tidy(() => {
                blobs.forEach((blob) => {
                    let a = document.createElement("a")
                    let url = URL.createObjectURL(blob)
                    let filename = `data.myz`
                    a.href = url
                    a.download = filename
                    a.click()
                    window.URL.revokeObjectURL(url)
                })
                blobs = []
            })
        }

        let maxPool = <nn.tfFn>((inp: tf.Tensor) => {
            return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, 5, 2, "same")
        })
        let prevCtrl1: number[] = new Array(7).fill(0)
        let prevCtrl2: number[] = new Array(7).fill(0)
        const loop = () => {
            count++
            if (getRestart()) {
                tf.tidy(() => {
                    let frames = tf.stack(clone(), 0)
                    let frameSave = { name: "frames", values: new Uint8Array(frames.dataSync()) }
                    let frameShape = { name: "frameShape", values: new Uint32Array(frames.shape) }

                    let ctrl1 = tf.stack(ctrl1Clone(), 0)
                    let ctrl1Save = { name: "ctrl1", values: new Uint8Array(ctrl1.dataSync()) }
                    let ctrl1Shape = { name: "ctrl1Shape", values: new Uint32Array(ctrl1.shape) }

                    let ctrl2 = tf.stack(ctrl2Clone(), 0)
                    let ctrl2Save = { name: "ctrl2", values: new Uint8Array(ctrl2.dataSync()) }
                    let ctrl2Shape = { name: "ctrl2Shape", values: new Uint32Array(ctrl2.shape) }

                    let blob = myz.save([frameSave, frameShape, ctrl1Save, ctrl1Shape, ctrl2Save, ctrl2Shape])

                    clear()
                    ctrl1Clear()
                    ctrl2Clear()
                    blobs.push(blob)
                })
            }
            next()
            render()

            let arrow1 = Math.floor(Math.random() * 8.99) + 1
            let attack1 = Math.floor(Math.random() * 3.99)
            const nowCtrl1 = [
                Number(arrow1 == 7 || arrow1 == 8 || arrow1 == 9),
                Number(arrow1 == 1 || arrow1 == 2 || arrow1 == 3),
                Number(arrow1 == 1 || arrow1 == 4 || arrow1 == 7),
                Number(arrow1 == 3 || arrow1 == 6 || arrow1 == 9),
                Number(attack1 == 1),
                Number(attack1 == 2),
                Number(attack1 == 3),
            ]
            ctrl1Write(tf.keep(tf.tensor((arrow1 - 1) * 4 + attack1)))

            let arrow2 = Math.floor(Math.random() * 8.99) + 1
            let attack2 = Math.floor(Math.random() * 3.99)
            const nowCtrl2 = [
                Number(arrow2 == 7 || arrow2 == 8 || arrow2 == 9),
                Number(arrow2 == 1 || arrow2 == 2 || arrow2 == 3),
                Number(arrow2 == 1 || arrow2 == 4 || arrow2 == 7),
                Number(arrow2 == 3 || arrow2 == 6 || arrow2 == 9),
                Number(attack2 == 1),
                Number(attack2 == 2),
                Number(attack2 == 3),
            ]
            ctrl2Write(tf.keep(tf.tensor((arrow2 - 1) * 4 + attack2)))

            control(
                {
                    arrow: {
                        jump: <1 | 0 | -1>(nowCtrl1[0] - prevCtrl1[0]),
                        squat: <1 | 0 | -1>(nowCtrl1[1] - prevCtrl1[1]),
                        left: <1 | 0 | -1>(nowCtrl1[2] - prevCtrl1[2]),
                        right: <1 | 0 | -1>(nowCtrl1[3] - prevCtrl1[3]),
                    },
                    attack: {
                        light: <1 | 0 | -1>(nowCtrl1[4] - prevCtrl1[4]),
                        medium: <1 | 0 | -1>(nowCtrl1[5] - prevCtrl1[5]),
                        heavy: <1 | 0 | -1>(nowCtrl1[6] - prevCtrl1[6]),
                    },
                },
                keySets[0]
            )
            prevCtrl1 = nowCtrl1
            control(
                {
                    arrow: {
                        jump: <1 | 0 | -1>(nowCtrl2[0] - prevCtrl2[0]),
                        squat: <1 | 0 | -1>(nowCtrl2[1] - prevCtrl2[1]),
                        left: <1 | 0 | -1>(nowCtrl2[2] - prevCtrl2[2]),
                        right: <1 | 0 | -1>(nowCtrl2[3] - prevCtrl2[3]),
                    },
                    attack: {
                        light: <1 | 0 | -1>(nowCtrl2[4] - prevCtrl2[4]),
                        medium: <1 | 0 | -1>(nowCtrl2[5] - prevCtrl2[5]),
                        heavy: <1 | 0 | -1>(nowCtrl2[6] - prevCtrl2[6]),
                    },
                },
                keySets[1]
            )
            prevCtrl2 = nowCtrl2

            tf.tidy(() => {
                let pix = <tf.Tensor3D>(
                    nn.pipe(
                        <nn.tfFn>((inp: tf.Tensor3D) => tf.image.resizeNearestNeighbor(inp, [256, 256])),
                        maxPool,
                        <nn.tfFn>((inp: tf.Tensor3D) => tf.image.resizeNearestNeighbor(inp, [32, 64])),
                        <nn.tfFn>((inp: tf.Tensor3D) => tf.cast(inp, "int32"))
                    )(<tf.Tensor3D>tf.browser.fromPixels(canvas))
                )

                tf.browser.toPixels(pix, pixcanvas)
                write(tf.keep(pix))
            })
            if (getRestart()) {
                console.log(count)
                count = 0
            }

            requestAnimationFrame(loop)
        }
        requestAnimationFrame(loop)
    })
