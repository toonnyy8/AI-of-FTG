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
    let posConv2 = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [config.r, 1, config.dmodel + config.ctrlNum * config.dact] }),
            tf.layers.separableConv2d({ filters: config.dmodel, kernelSize: [config.r, 1], padding: "valid" }),
        ],
    })

    let mha1 = MHA(config.dmodel, config.head, config.dk, config.dv)
    let ff1 = FF(config.dmodel, config.dmodel * 2)

    let memNum = 64
    let memSize = 200
    let extMem = tf.variable(tf.randomNormal([1, memNum, memSize]), true, "extMem")

    let mha2 = MHA(config.dmodel, config.head, config.dk, config.dv)
    let ff2 = FF(config.dmodel, config.dmodel * 2)
    let ff3 = FF(config.dmodel, config.dmodel * 2)

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

                let pos1Out = <tf.Tensor4D>(<tf.Tensor4D>posConv1.apply(inp)) //.reshape([1, l, h * w * c])
                pos1Out = unk
                    .reshape([1, 1, 1, -1])
                    .tile([1, config.r - 1, 1, 1])
                    .concat(tf.concat([pos1Out, ...embs], -1), 1)
                let pos2Out = <tf.Tensor3D>(<tf.Tensor4D>posConv2.apply(pos1Out)).reshape([1, l, h * w * c])

                let mha1Out = <tf.Tensor3D>mha1.fn(pos2Out, pos2Out)
                mha1Out = mha1Out.add(pos2Out)
                let ff1Out = nn.mish(<tf.Tensor3D>ff1.fn(mha1Out))
                ff1Out = ff1Out.add(mha1Out)

                let mha2Out = <tf.Tensor3D>mha2.fn(ff1Out, ff1Out)
                mha2Out = mha2Out.add(ff1Out)
                let ff2Out = nn.mish(<tf.Tensor3D>ff2.fn(mha2Out))
                ff2Out = ff2Out.add(mha2Out)

                let out = <tf.Tensor3D>ff3.fn(ff2Out)

                return out.reshape([l, h, w, c])
            }),
        ws: () =>
            tf.tidy(() => [
                ...actEmbs,
                unk,
                ...(<tf.Variable[]>posConv1.getWeights()),
                ...(<tf.Variable[]>posConv2.getWeights()),
                ...mha1.ws(),
                ...ff1.ws(),
                // extMem,
                ...mha2.ws(),
                ...ff2.ws(),
                ...ff3.ws(),
            ]),
        unk: () => {
            return unk
        },
    }
}
