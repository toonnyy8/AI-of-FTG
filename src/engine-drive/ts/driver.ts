import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"
import { MHA, FF, positionalEncoding } from "./mha"

export const Driver = (config: {
    ctrlNum: number
    actionNum: number
    dinp: number
    dmodel: number
    dact?: number
    r?: number
    head?: number
    dk?: number
    dv?: number
    hiddens?: number
}): {
    fn: (input: tf.Tensor2D, actions: number[][]) => tf.Tensor2D
    ws: () => tf.Variable[]
} => {
    const ctrlNum = config.ctrlNum
    const actionNum = config.actionNum
    const dinp = config.dinp
    const dmodel = config.dmodel
    const dact = config.dact !== undefined ? config.dact : 32
    const r = config.r !== undefined ? config.r : 8
    const head = config.head !== undefined ? config.head : 8
    const dk = config.dk !== undefined ? config.dk : 32
    const dv = config.dv !== undefined ? config.dv : 32
    const hiddens = config.hiddens !== undefined ? config.hiddens : dmodel * 2

    let actEmbs = new Array(ctrlNum)
        .fill(0)
        .map((_, idx) => tf.variable(tf.randomNormal([actionNum, dmodel]), true, `actEmb${idx}`))

    let inpLinear = tf.sequential({
        layers: [tf.layers.inputLayer({ inputShape: [dinp] }), tf.layers.dense({ units: dmodel })],
    })

    let mha1 = MHA(dmodel, head, dk, dv)
    let ff1 = FF(dmodel, hiddens)

    let mha2 = MHA(dmodel, head, dk, dv)
    let ff2 = FF(dmodel, hiddens)

    let mha3 = MHA(dmodel, head, dk, dv)
    let ff3 = FF(dmodel, hiddens)

    let outLinear = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [dmodel] }),
            tf.layers.dense({ units: hiddens }),
            layers.mish({}),
            tf.layers.dense({ units: dinp, activation: "tanh" }),
        ],
    })

    const norm = <T extends tf.Tensor>(x: T) =>
        tf.tidy(() => {
            let mean = x.mean()
            let variance = x.sub(mean).square().mean()
            return <T>x.sub(mean).div(variance.add(0.001).sqrt())
        })

    let posEnc: tf.Tensor2D[] = []

    return {
        // ctrlActions:number[ctrls][len]
        fn: (input: tf.Tensor2D, ctrlActions: number[][]) =>
            tf.tidy(() => {
                let [l, c] = input.shape
                if (dinp != c) throw new Error("c != dinp")
                if (posEnc[l] === undefined) posEnc[l] = tf.keep(positionalEncoding(l, dmodel))
                let embs: tf.Variable<tf.Rank.R2>[] = ctrlActions.map((actions, idx) => {
                    return <tf.Variable<tf.Rank.R2>>tf.gather(actEmbs[idx], actions, 0)
                })

                let st = <tf.Tensor2D>nn.mish(<tf.Tensor2D>inpLinear.apply(input))
                st = tf.addN([st, posEnc[l]])
                let act = tf.addN([...embs, posEnc[l]])

                // 嵌入 act
                let stAct = <tf.Tensor2D>mha1.fn(st, act)
                stAct = tf.addN([stAct, st, act])
                let ff1Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff1.fn(stAct))
                ff1Out = ff1Out.add(stAct)

                let mha2Out = <tf.Tensor2D>mha2.fn(ff1Out, ff1Out)
                mha2Out = mha2Out.add(ff1Out)
                let ff2Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff2.fn(mha2Out))
                ff2Out = ff2Out.add(mha2Out)

                // let mha3Out = <tf.Tensor2D>mha3.fn(ff2Out, ff2Out)
                // mha3Out = mha3Out.add(ff2Out)
                // let ff3Out = <tf.Tensor2D>nn.mish(<tf.Tensor2D>ff3.fn(mha3Out))
                // ff3Out = ff3Out.add(mha3Out)

                let out = <tf.Tensor2D>outLinear.apply(ff2Out)

                return out
            }),
        ws: () =>
            tf.tidy(() => [
                ...actEmbs,
                ...(<tf.Variable[]>inpLinear.getWeights()),
                ...mha1.ws(),
                ...ff1.ws(),
                ...mha2.ws(),
                ...ff2.ws(),
                ...mha3.ws(),
                ...ff3.ws(),
                ...(<tf.Variable[]>outLinear.getWeights()),
            ]),
    }
}
