import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"
import { transformer } from "./model"

export function trainFn() {
    return tf.tidy(() => {
        let loss, newMems
        let allVars = tfex.scope.variableScope("transformerXL").trainableVariables()
        let grads = tf.grads(() => {
            [loss, newMems] = transformer({
                decInp: inp,
                target: tgt,
                mems: mems,
                nToken: n_token,
                nLayer: FLAGS.n_layer,
                dModel: FLAGS.d_model,
                dEmbed: FLAGS.d_embed,
                nHead: FLAGS.n_head,
                dHead: FLAGS.d_head,
                dInner: FLAGS.d_inner,
                dropout: FLAGS.dropout,
                dropatt: FLAGS.dropatt,
                initializer: initializer,
                projInitializer: proj_initializer,
                isTraining: is_training,
                memLen: FLAGS.mem_len,
                cutoffs: cutoffs,
                divVal: FLAGS.div_val,
                tieProjs: tie_projs,
                inputPerms: None,
                targetPerms: None,
                headTarget: None,
                sameLength: FLAGS.same_length,
                clampLen: FLAGS.clamp_len,
                useTpu: False,
                untieR: FLAGS.untie_r,
                projSameDim: FLAGS.proj_same_dim
            },
                tfex.scope.variableScope("transformerXL")
            )
            tf.keep(newMems)
            return loss
        })(allVars)

        return [
            newMems,
            allVars.reduce((last, v, idx) => {
                last[v.name] = grads[idx]
            }, {})
        ]
    })
}

export function t() {
    let mems = null
    let towerNamedGrads = []
    for (; ;) {
        let [newMems, namedGrads] = trainFn(mems)

        tf.dispose(mems)
        mems = newMems
        towerNamedGrads.push(namedGrads)
    }
    gradsAndVars = averageGradsAndVars(towerGradsAndVars)
    clipped, gnorm = tf.clipByValue(grads, FLAGS.clip)
    grads_and_vars = list(zip(clipped, all_vars))
    tf.train.adam().applyGradients()
}