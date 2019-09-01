import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"
import { transformer } from "./model"

export function modelFn(inp, tgt, nToken, FLAGS, initializer, projInitializer, isTraining = true) {
    return tf.tidy(() => {
        let allVars = tfex.scope.variableScope("transformerXL").trainableVariables()
        let towerNamedGrads = allVars.reduce((last, v) => {
            last[v.name] = []
            return last
        }, {})
        let outputs = []
        let mems = null

        for (let i = 0; i < 1; i++) {
            let grads = tf.grads(() => {
                let [loss, newMems, output] = transformer({
                    decInp: inp,
                    target: tgt,
                    mems: mems,
                    nToken: nToken,//113
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
                return loss
            })(allVars)

            Object.keys(towerNamedGrads).forEach((name, idx) => {
                towerNamedGrads[name].push(grads[idx])
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
                    towerNamedGrads.length
                )
            })
            return last
        }, {})
    })
}

export function train() {
    return tf.tidy(() => {

        let [outputs, towerNamedGrads] = modelFn()

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
        tf.train.adam().applyGradients(
            names.reduce((last, name, idx) => {
                last[name] = clipped[idx]
            }, {})
        )

        tf.dispose(outputs)
        Object.keys(towerNamedGrads).forEach((name) => {
            tf.dispose(towerNamedGrads[name])
        })
    })

}