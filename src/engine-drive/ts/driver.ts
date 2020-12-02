import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"
import { MHA, FF } from "./mha"
import { sequential } from "@tensorflow/tfjs"

export const Driver = (config: {
    ctrlNum: number
    actionNum: number
    dmodel: number
    dact?: number
    r?: number
    head?: number
    dk?: number
    dv?: number
    hiddens?: number
}): {
    fn: (input: tf.Tensor4D, actions: number[][]) => tf.Tensor
    ws: () => tf.Variable[]
    unk: () => tf.Variable
} => {
    const ctrlNum = config.ctrlNum
    const actionNum = config.actionNum
    const dmodel = config.dmodel
    const dact = config.dact !== undefined ? config.dact : 32
    const r = config.r !== undefined ? config.r : 8
    const head = config.head !== undefined ? config.head : 8
    const dk = config.dk !== undefined ? config.dk : 32
    const dv = config.dv !== undefined ? config.dv : 32
    const hiddens = config.hiddens !== undefined ? config.hiddens : 400

    let actEmbs = new Array(ctrlNum)
        .fill(0)
        .map((_, idx) => tf.variable(tf.randomNormal([actionNum, dact]), true, `actEmb${idx}`))
    let unk = tf.variable(tf.randomNormal([dmodel + ctrlNum * dact]), true, "unk")

    let posConv1 = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [r, 1, dmodel + ctrlNum * dact] }),
            tf.layers.separableConv2d({ filters: dmodel, kernelSize: [r, 1], padding: "valid" }),
        ],
    })

    let mha1 = MHA(dmodel, head, dk, dv)
    let ff1 = FF(dmodel, hiddens)

    let mha2 = MHA(dmodel, head, dk, dv)
    let ff2 = FF(dmodel, hiddens)

    let ffout = FF(dmodel, hiddens)

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
                if (dmodel != h * w * c) throw new Error("h*w*c != dmodel")
                let embs: tf.Variable<tf.Rank.R4>[] = ctrlActions.map((actions, idx) => {
                    return <tf.Variable<tf.Rank.R4>>tf.gather(actEmbs[idx], actions, 0).reshape([1, l, 1, dact])
                })
                let inp = unk
                    .reshape([1, 1, 1, -1])
                    .tile([1, r - 1, 1, 1])
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

                let out = <tf.Tensor2D>ffout.fn(ff2Out)

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
                ...ffout.ws(),
            ]),
        unk: () => {
            return unk
        },
    }
}
