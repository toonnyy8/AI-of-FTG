import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

import * as tf from "@tensorflow/tfjs"
import { AED } from "../model/ae"
import * as nn from "../model/nn"

tf.setBackend("webgl")

const keySets: [
    {
        jump: string;
        squat: string;
        left: string;
        right: string;
        attack: {
            light: string;
            medium: string;
            heavy: string;
        };
    },
    {
        jump: string;
        squat: string;
        left: string;
        right: string;
        attack: {
            light: string;
            medium: string;
            heavy: string;
        };
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

let d1canvas = <HTMLCanvasElement>document.getElementById("d1Canvas")
let d2canvas = <HTMLCanvasElement>document.getElementById("d2Canvas")
let d3canvas = <HTMLCanvasElement>document.getElementById("d3Canvas")

let control = (ctrl, keySet) => {
    switch (ctrl) {
        case 0: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["jump"]
                })
            )
            break;
        }
        case 1: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["squat"]
                })
            )
            break;
        }
        case 2: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["left"]
                })
            )
            break;
        }
        case 3: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["right"]
                })
            )
            break;
        }
        case 4: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["attack"]["light"]
                })
            )
            break;
        }
        case 5: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["attack"]["medium"]
                })
            )
            break;
        }
        case 6: {
            document.dispatchEvent(
                new KeyboardEvent("keydown", {
                    key: keySet["attack"]["heavy"]
                })
            )
            break;
        }
        case 7: {
            break;
        }
        case 8: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["jump"]
                })
            )
            if (ctrl == 7) break;
        }
        case 9: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["squat"]
                })
            )
            if (ctrl == 7) break;;
        }
        case 10: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["left"]
                })
            )
            if (ctrl == 7) break;;
        }
        case 11: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["right"]
                })
            )
            if (ctrl == 7) break;;
        }
        case 12: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["light"]
                })
            )
            if (ctrl == 7) break;;
        }
        case 13: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["medium"]
                })
            )
            if (ctrl == 7) break;;
        }
        case 14: {
            document.dispatchEvent(
                new KeyboardEvent("keyup", {
                    key: keySet["attack"]["heavy"]
                })
            )
            if (ctrl == 7) break;;
        }
    }
}

let op = tf.train.adamax(2e-4)
{
    (async () => {
        let [{ fn: ae1, ws: ae1_ws }, { fn: ad1, ws: ad1_ws }] = AED([3, 32, 32])
        let [{ fn: ae2, ws: ae2_ws }, { fn: ad2, ws: ad2_ws }] = AED([32, 128, 128])
        let [{ fn: ae3, ws: ae3_ws }, { fn: ad3, ws: ad3_ws }] = AED([128, 512, 512])
        let { next, getP1, getP2, getRestart } = await Game(
            keySets,
            canvas,
        )
        let count = 0
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
                tf.tidy(() => {
                    op.minimize(() => {
                        let pix = tf
                            .image
                            .resizeNearestNeighbor(tf.browser.fromPixels(canvas), [64, 64])
                            .cast("float32")
                            .div(255);
                        let b = tf.stack([pix, pix.reverse([1, 2])], 0)
                        let e1 = ae1(b)
                        let e2 = ae2(e1)
                        let e3 = ae3(e2)
                        let d1 = ad1(e1)
                        let d2 = ad1(nn.mish(ad2(e2)))
                        let d3 = ad1(nn.mish(ad2(nn.mish(ad3(e3)))))

                        let loss1_1 = tf.losses.sigmoidCrossEntropy(b, d1)
                        let loss2_1 = tf.losses.sigmoidCrossEntropy(b, d2)
                        let loss3_1 = tf.losses.sigmoidCrossEntropy(b, d3)

                        tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d1.unstack(0)[0]), d1canvas)
                        tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d2.unstack(0)[0]), d2canvas)
                        tf.browser.toPixels(tf.sigmoid(<tf.Tensor3D>d3.unstack(0)[0]), d3canvas)
                        let loss = tf.addN([loss1_1, loss2_1.mul(2), loss3_1.mul(3)]).div(6)
                        loss.print()
                        return <tf.Scalar>loss
                    },
                        false,
                        [...ae1_ws(), ...ae2_ws(), ...ae3_ws(), ...ad3_ws(), ...ad2_ws(), ...ad1_ws()])
                    // let pix = tf
                    //     .image
                    //     .resizeNearestNeighbor(tf.browser.fromPixels(canvas), [64, 64])
                    //     .cast("float32")
                    //     .div(255);
                    // tf.browser.toPixels(<tf.Tensor3D>pix, p1canvas)
                })
            }

            requestAnimationFrame(loop)
        }
        requestAnimationFrame(loop)

    })()
}
