import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as imgz from "./imgz"

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

let control = (ctrl: number, keySet: KeySet) => {
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
        }
        case 9: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["squat"],
                })
            )
        }
        case 10: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["left"],
                })
            )
        }
        case 11: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["right"],
                })
            )
        }
        case 12: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["light"],
                })
            )
        }
        case 13: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["medium"],
                })
            )
        }
        case 14: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["heavy"],
                })
            )
        }
    }
}

const _control = (ctrl: {
    jump: boolean,
    squat: boolean,
    left: boolean,
    right: boolean,
    attack: {
        light: boolean,
        medium: boolean,
        heavy: boolean,
    },
}, keySet: KeySet) => {
    Object.keys(ctrl).map((ctrlName) => {
        if (ctrlName == "attack") {
            Object.keys(ctrl.attack).map(atkName => {
                if (
                    ctrl.attack[<"light" | "medium" | "heavy">atkName]
                ) {
                    document.dispatchEvent(
                        new KeyboardEvent("keydown", {
                            key: keySet.attack[<"light" | "medium" | "heavy">atkName],
                        })
                    )
                } else {
                    document.dispatchEvent(
                        new KeyboardEvent("keyup", {
                            key: keySet.attack[<"light" | "medium" | "heavy">atkName],
                        })
                    )
                }
            })
        }
        else {
            if (
                ctrl[<"jump" | "squat" | "left" | "right">ctrlName]
            ) {
                document.dispatchEvent(
                    new KeyboardEvent("keydown", {
                        key: keySet[<"jump" | "squat" | "left" | "right">ctrlName],
                    })
                )
            } else {
                document.dispatchEvent(
                    new KeyboardEvent("keyup", {
                        key: keySet[<"jump" | "squat" | "left" | "right">ctrlName],
                    })
                )
            }
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

            ; (<HTMLButtonElement>document.getElementById("save")).onclick = () => {
                tf.tidy(() => {
                    let tensors = clone()
                    let frames = { name: "frames", tensor: tf.stack(tensors, 0) }

                    let blob = imgz.save([frames])
                    let a = document.createElement("a")
                    let url = URL.createObjectURL(blob)
                    let filename = `${tensors.length}.imgz`
                    a.href = url
                    a.download = filename
                    a.click()
                    window.URL.revokeObjectURL(url)
                    clear()
                })
            }

        let maxPool = <nn.tfFn>((inp: tf.Tensor) => {
            return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, 5, 2, "same")
        })
        let maxPool3 = <nn.tfFn>((inp: tf.Tensor) => {
            return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, 3, 2, "same")
        })
        let maxPool_1_2 = <nn.tfFn>((inp: tf.Tensor) => {
            return <tf.Tensor>tf.maxPool(<tf.Tensor4D>inp, [1, 2], [1, 2], "same")
        })
        let avgPool2 = <nn.tfFn>((inp: tf.Tensor) => {
            return <tf.Tensor>tf.avgPool(<tf.Tensor4D>inp, 2, 2, "same")
        })
        let blurPooling = nn.blurPooling(7, 2)
        let blurPooling3 = nn.blurPooling(3, 2)
        let blurPooling5 = nn.blurPooling(5, 2)
        const loop = () => {
            count++
            next()
            render()
            tf.tidy(() => {
                let pix = <tf.Tensor3D>(
                    nn.pipe(
                        // <nn.tfFn>((inp: tf.Tensor3D) => tf.image.resizeBilinear(inp, [256, 256])),
                        maxPool,
                        <nn.tfFn>blurPooling3.fn,
                        maxPool3,
                        <nn.tfFn>blurPooling3.fn,
                        // <nn.tfFn>blurPooling3.fn,
                        <nn.tfFn>((inp: tf.Tensor3D) => tf.image.resizeBilinear(inp, [32, 32])),
                        <nn.tfFn>((inp: tf.Tensor3D) => tf.cast(inp, "int32")),
                    )(<tf.Tensor3D>tf.browser.fromPixels(canvas))
                )

                tf.browser.toPixels(pix, pixcanvas)
                write(tf.keep(pix))
            })
            if (getRestart()) {
                console.log(count)
                count = 0
            } else {
                const ctrl1 = {
                    jump: Math.random() < 0.3 ? true : false,
                    squat: Math.random() < 0.1 ? true : false,
                    left: Math.random() < 0.5 ? true : false,
                    right: Math.random() < 0.5 ? true : false,
                    attack: {
                        light: Math.random() < 0.01 ? true : false,
                        medium: Math.random() < 0.01 ? true : false,
                        heavy: Math.random() < 0.01 ? true : false,
                    },
                }
                const ctrl2 = {
                    jump: Math.random() < 0.2 ? true : false,
                    squat: Math.random() < 0.2 ? true : false,
                    left: Math.random() < 0.4 ? true : false,
                    right: Math.random() < 0.4 ? true : false,
                    attack: {
                        light: Math.random() < 0.1 ? true : false,
                        medium: Math.random() < 0.1 ? true : false,
                        heavy: Math.random() < 0.1 ? true : false,
                    },
                }
                // _control(ctrl1, keySets[0])
                _control(ctrl2, keySets[1])
            }


            requestAnimationFrame(loop)
        }
        requestAnimationFrame(loop)
    })
