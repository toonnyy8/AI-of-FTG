import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"
import { MHA, FF } from "./mha"
import { sequential } from "@tensorflow/tfjs"

export const Driver = (
    config: {
        ctrlNum: number
        actionNum: number
        dact: number
        dmodel: number
        r: number
        head: number
        dk: number
        dv: number
    } = { ctrlNum: 2, actionNum: 2 ** 7, dact: 32, dmodel: 200, r: 8, head: 8, dk: 32, dv: 32 }
): {
    fn: (input: tf.Tensor4D, actions: number[][]) => tf.Tensor
    ws: () => tf.Variable[]
    unk: () => tf.Variable
} => {
    let actEmbs = new Array(config.ctrlNum)
        .fill(0)
        .map((_, idx) => tf.variable(tf.randomNormal([config.actionNum, config.dact]), true, `actEmb${idx}`))
    let unk = tf.variable(tf.randomNormal([config.dmodel + config.ctrlNum * config.dact]), true, "unk")

    let posConv1 = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [config.r, 1, config.dmodel + config.ctrlNum * config.dact] }),
            tf.layers.separableConv2d({ filters: config.dmodel, kernelSize: [config.r, 1], padding: "valid" }),
        ],
    })

    let mha1 = MHA(config.dmodel, config.head, config.dk, config.dv)
    let ff1 = FF(config.dmodel, config.dmodel * 2)

    let mha2 = MHA(config.dmodel, config.head, config.dk, config.dv)
    let ff2 = FF(config.dmodel, config.dmodel * 2)

    // let mha3 = MHA(config.dmodel, config.head, config.dk, config.dv)
    // let ff3 = FF(config.dmodel, config.dmodel * 2)

    // let mha4 = MHA(config.dmodel, config.head, config.dk, config.dv)
    // let ff4 = FF(config.dmodel, config.dmodel * 2)

    let ff5 = FF(config.dmodel, config.dmodel * 2)

    const norm = <T extends tf.Tensor>(x: T) =>
        tf.tidy(() => {
            let mean = x.mean()
            let variance = x.sub(mean).square().mean()
            return <T>x.sub(mean).div(variance.add(0.001).sqrt())
        })

    return {
        // ctrlActions:number[ctrls][len]
        fn: (input: tf.Tensor4D, ctrlActions: number[][]) =>
            tf.tidy(() => {
                let [l, h, w, c] = input.shape
                if (config.dmodel != h * w * c) throw new Error("h*w*c != dmodel")
                let embs: tf.Variable<tf.Rank.R4>[] = ctrlActions.map((actions, idx) => {
                    return <tf.Variable<tf.Rank.R4>>tf.gather(actEmbs[idx], actions, 0).reshape([1, l, 1, config.dact])
                })
                let inp = unk
                    .reshape([1, 1, 1, -1])
                    .tile([1, config.r - 1, 1, 1])
                    .concat(tf.concat([<tf.Tensor4D>input.reshape([1, l, 1, h * w * c]), ...embs], -1), 1)

                let pos1Out = <tf.Tensor2D>(<tf.Tensor4D>posConv1.apply(inp)).reshape([l, h * w * c])

                let mha1Out = <tf.Tensor2D>mha1.fn(pos1Out, pos1Out)
                mha1Out = mha1Out.add(pos1Out)
                let ff1Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff1.fn(mha1Out))
                ff1Out = ff1Out.add(mha1Out)

                let mha2Out = <tf.Tensor2D>mha2.fn(ff1Out, ff1Out)
                mha2Out = mha2Out.add(ff1Out)
                let ff2Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff2.fn(mha2Out))
                ff2Out = ff2Out.add(mha2Out)

                // let mha3Out = <tf.Tensor2D>mha3.fn(ff2Out, ff2Out)
                // mha3Out = norm(mha3Out.add(ff2Out))
                // let ff3Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff3.fn(mha3Out))
                // ff3Out = norm(ff3Out.add(mha3Out))

                // let mha4Out = <tf.Tensor2D>mha4.fn(ff3Out, ff3Out)
                // mha4Out = norm(mha4Out.add(ff3Out))
                // let ff4Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff4.fn(mha4Out))
                // ff4Out = norm(ff4Out.add(mha4Out))

                let out = <tf.Tensor2D>ff5.fn(ff2Out)

                return out.reshape([l, h, w, c])
            }),
        ws: () =>
            tf.tidy(() => [
                ...actEmbs,
                unk,
                ...(<tf.Variable[]>posConv1.getWeights()),
                ...mha1.ws(),
                ...ff1.ws(),
                ...mha2.ws(),
                ...ff2.ws(),
                // ...mha3.ws(),
                // ...ff3.ws(),
                // ...mha4.ws(),
                // ...ff4.ws(),
                ...ff5.ws(),
            ]),
        unk: () => {
            return unk
        },
    }
}
