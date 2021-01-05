import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"
import * as layers from "./layers"
import { MHA, FF, positionalEncoding } from "./mha"

export const Driver = (config: {
    ctrlNum: number
    actionNum: number
    dinp: number
    dmodel: number
    head?: number
    dk?: number
    dv?: number
    hiddens?: number
    restrictHead?: number
    layerNum?: number
}): {
    fn: (input: tf.Tensor2D, actions: number[][]) => tf.Tensor2D
    ws: () => tf.Variable[]
} => {
    const ctrlNum = config.ctrlNum
    const actionNum = config.actionNum
    const dinp = config.dinp
    const dmodel = config.dmodel
    const head = config.head !== undefined ? config.head : 8
    const dk = config.dk !== undefined ? config.dk : 32
    const dv = config.dv !== undefined ? config.dv : 32
    const hiddens = config.hiddens !== undefined ? config.hiddens : dmodel * 2
    const restrictHead = config.restrictHead !== undefined ? config.restrictHead : 4
    const layerNum = config.layerNum !== undefined ? config.layerNum : 3

    let actEmbs = new Array(ctrlNum)
        .fill(0)
        .map((_, idx) => tf.variable(tf.randomNormal([actionNum, dmodel]), true, `actEmb${idx}`))

    let inpLinear = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [dinp] }),
            tf.layers.dense({ name: "inpLinear", units: dmodel }),
            layers.mish({}),
        ],
    })

    let mha_ff = new Array(layerNum).fill(0).map((_, idx) => {
        return { mha: MHA(dmodel, head, dk, dv, `mha${idx + 1}`), ff: FF(dmodel, hiddens, `ff${idx + 1}`) }
    })

    let outLinear = tf.sequential({
        layers: [
            tf.layers.inputLayer({ inputShape: [dmodel] }),
            tf.layers.dense({ name: "outLinear0", units: hiddens }),
            layers.mish({}),
            tf.layers.dense({ name: "outLinear1", units: dinp }),
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

                let inp = <tf.Tensor2D>inpLinear.apply(input)
                inp = tf.addN([inp, ...embs, posEnc[l]])

                let out = mha_ff.reduce((inp, curr) => {
                    let mhaOut = curr.mha.fn(inp, inp)
                    mhaOut = nn.mish(mhaOut)
                    mhaOut = mhaOut.add(inp)
                    let ffOut = curr.ff.fn(mhaOut)
                    ffOut = nn.mish(ffOut)
                    ffOut = ffOut.add(mhaOut)
                    return ffOut
                }, inp)

                out = <tf.Tensor2D>outLinear.apply(out)

                return out.tanh()
            }),
        ws: () =>
            tf.tidy(() => [
                ...actEmbs,
                ...(<tf.Variable[]>inpLinear.getWeights()),
                ...mha_ff.reduce((ws, curr) => {
                    return [...ws, ...curr.mha.ws(), ...curr.ff.ws()]
                }, <tf.Variable[]>[]),
                ...(<tf.Variable[]>outLinear.getWeights()),
            ]),
    }
}