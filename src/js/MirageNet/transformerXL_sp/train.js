import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../../lib/tfjs-extensions/src"
import { transformer } from "./model"
const tfex = registerTfex(tf)

let optimizer = tf.train.adam(0.0005)

export function modelFn(inp, tgt, nToken, FLAGS, initializer, projInitializer, isTraining = true) {
    return tf.tidy(() => {
        let outputs = []
        let mems = null

        for (let i = 0; i < inp.length; i++) {
            tf.tidy(() => {
                let [output, newMems] = transformer({
                    decInp: inp[i],
                    target: tgt[i],
                    mems: mems,
                    nToken: nToken, //1384
                    nLayer: FLAGS.nLayer,
                    dModel: FLAGS.dModel,
                    dEmbed: FLAGS.dEmbed,
                    nHead: FLAGS.nHead,
                    dHead: FLAGS.dHead,
                    dInner: FLAGS.dInner,
                    dropout: FLAGS.dropout,
                    dropatt: FLAGS.dropatt,
                    initializer: initializer,
                    projInitializer: projInitializer,
                    isTraining: isTraining,
                    memLen: FLAGS.memLen,
                    cutoffs: [],
                    divVal: FLAGS.divVal,
                    tieProjs: [],
                    inputPerms: null,
                    targetPerms: null,
                    headTarget: null,
                    sameLength: FLAGS.sameLength,
                    clampLen: FLAGS.clampLen,
                    useTpu: false,
                    untieR: FLAGS.untieR,
                    projSameDim: FLAGS.projSameDim
                },
                    tfex.scope.variableScope("transformerXL")
                )

                tf.dispose(mems)
                mems = tf.keep(newMems)
                outputs.push(tf.keep(output))
            })
        }

        tf.dispose(mems)

        return outputs
    })
}

export function gradModelFn(inp, tgt, nToken, FLAGS, initializer, projInitializer, isTraining = true) {
    return tf.tidy(() => {
        let allVars = tfex.scope.variableScope("transformerXL").trainableVariables()
        let towerNamedGrads = allVars.reduce((last, v) => {
            last[v.name] = []
            return last
        }, {})
        let outputs = []
        let mems = null

        for (let i = 0; i < inp.length; i++) {
            let grads = optimizer.computeGradients(() => {
                let [output, newMems] = transformer({
                    decInp: inp[i],
                    target: tgt[i],
                    mems: mems,
                    nToken: nToken, //1496
                    nLayer: FLAGS.nLayer,
                    dModel: FLAGS.dModel,
                    dEmbed: FLAGS.dEmbed,
                    nHead: FLAGS.nHead,
                    dHead: FLAGS.dHead,
                    dInner: FLAGS.dInner,
                    dropout: FLAGS.dropout,
                    dropatt: FLAGS.dropatt,
                    initializer: initializer,
                    projInitializer: projInitializer,
                    isTraining: isTraining,
                    memLen: FLAGS.memLen,
                    cutoffs: [],
                    divVal: FLAGS.divVal,
                    tieProjs: [],
                    inputPerms: null,
                    targetPerms: null,
                    headTarget: null,
                    sameLength: FLAGS.sameLength,
                    clampLen: FLAGS.clampLen,
                    useTpu: false,
                    untieR: FLAGS.untieR,
                    projSameDim: FLAGS.projSameDim
                },
                    tfex.scope.variableScope("transformerXL")
                )

                tf.dispose(mems)
                mems = tf.keep(newMems)
                outputs.push(tf.keep(output))

                let loss = ((logits, labels, dim) => {
                    return tf.tidy(() => {
                        let h = tf.mul(-1, tf.mul(labels, tf.log(logits)).sum(dim))
                        return tf.mean(h)
                    })
                })(output, tf.oneHot(tf.cast(tgt[i], "int32"), output.shape[2]), 2)

                loss.print()
                return loss
            }, allVars).grads

            Object.keys(towerNamedGrads).forEach((name) => {
                grads[name].isNaN().any().array().then(a => {
                    if (a) {
                        console.log(name)
                    }
                })
                towerNamedGrads[name].push(grads[name])
            })
        }

        tf.dispose(mems)

        return [outputs, towerNamedGrads]
    })
}

function averageNamedGrads(towerNamedGrads) {
    return tf.tidy(() => {
        return Object.keys(towerNamedGrads).reduce((last, name) => {
            last[name] = tf.tidy(() => {
                return tf.div(
                    tf.addN(towerNamedGrads[name]),
                    towerNamedGrads[name].length
                )
            })
            return last
        }, {})
    })
}

export function train(inp, tgt, nToken, FLAGS, initializer, projInitializer, isTraining = true, bsz = 1) {
    return tf.tidy(() => {
        let batch = (ts) => {
            let output = tf.stack(ts)
            return output.split(output.shape[2] / bsz, 2).map((x) => {
                return tf.unstack(x)
            })
        }
        let inps = batch(inp)
        let tgts = batch(tgt)
        let batchNamedGrads = []
        for (let i = 0; i < inps.length; i++) {
            let [outputs, towerNamedGrads] = gradModelFn(inps[i], tgts[i], nToken, FLAGS, initializer, projInitializer, isTraining)
            batchNamedGrads.push(averageNamedGrads(towerNamedGrads))
            tf.dispose(outputs)
        }

        let namedGrads = averageNamedGrads(
            Object.keys(batchNamedGrads[0]).reduce((last, name) => {
                last[name] = batchNamedGrads.map((bng) => bng[name])
                return last
            }, {})
        )

        let [names, grads] = Object.keys(namedGrads).reduce((last, name) => {
            last[0].push(name)
            last[1].push(namedGrads[name])
            return last
        }, [
            [],
            []
        ])
        let [clipped, gnorm] = tfex.funcs.clipByGlobalNorm(grads, FLAGS.clip)
        optimizer.applyGradients(
            names.reduce((last, name, idx) => {
                last[name] = clipped[idx]
                return last
            }, {})
        )
    })
}