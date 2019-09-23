import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"
import { transformer } from "./model"

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
                    nToken: nToken, //113
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
            // console.log(tf.memory())
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
                    nToken: nToken, //113
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
                // console.log("output")
                // output.isNaN().any().print()

                let loss = ((logits, labels, dim) => {
                    return tf.tidy(() => {
                        // y = tf.clipByValue(y, 0.001, 1)
                        let h = tf.mul(-1, tf.mul(labels, tf.log(logits)).sum(dim))
                        return tf.mean(h)
                    })
                })(output, tf.oneHot(tf.cast(tgt[i], "int32"), output.shape[2]), 2)

                loss.print()
                return loss
            }, allVars).grads
            // Object.values(grads).forEach(g => g.print())
            Object.keys(towerNamedGrads).forEach((name) => {
                towerNamedGrads[name].push(grads[name])
                // console.log(name)
                // grads[name].isNaN().any().print()
            })
        }

        // allVars.forEach(v => {
        //     console.log(v.name)
        //     v.isNaN().any().print()
        // })

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

export function train(inp, tgt, nToken, FLAGS, initializer, projInitializer, isTraining = true) {
    return tf.tidy(() => {

        let [outputs, towerNamedGrads] = gradModelFn(inp, tgt, nToken, FLAGS, initializer, projInitializer, isTraining)
        tf.dispose(outputs)

        let namedGrads = averageNamedGrads(towerNamedGrads)
        let [names, grads] = Object.keys(namedGrads).reduce((last, name) => {
            last[0].push(name)
            last[1].push(namedGrads[name])
            return last
        }, [
            [],
            []
        ])
        let [clipped, gnorm] = tfex.clipByGlobalNorm(grads, FLAGS.clip)
        optimizer.applyGradients(
            names.reduce((last, name, idx) => {
                last[name] = clipped[idx]
                // console.log(name)
                // clipped[idx].print()
                return last
            }, {})
        )

        Object.keys(towerNamedGrads).forEach((name) => {
            tf.dispose(towerNamedGrads[name])
        })

    })

}