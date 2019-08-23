import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"

function myDense(
    args = {
        x,
        units,
        activation,
        kernelInitializer,
        name: "",
        useBias: true
    },
    scope = tfex.scope
) {
    return tf.tidy(() => {
        let output = tf.dot(x,
            scope.getVariable(`${name}_kernel`, x.concat([units]), true, "float32"))

        if (useBias) {
            output = tf.add(output,
                scope.getVariable(`${name}_bias`, x.concat([units]), true, "float32")
            )
        }

        if (activation != null) {
            output = tf[activation](output)
        }

        return output

    })
}

export function positionalEmbedding(
    args = {
        bsz: null,
        posSeq,
        invFreq
    }
) {
    return tf.tidy(() => {
        let sinusoidInp = tfex.einsum('i,j->ij', args.posSeq, args.invFreq)
        let posEmb = tf.concat([tf.sin(sinusoidInp), tf.cos(sinusoidInp)], -1)
        if (args.bsz != null) {
            return posEmb.expandDims(1).tile([1, args.bsz, 1])
        } else {
            return posEmb.expandDims(1)
        }
    })

}

export function positionwiseFF(
    args = {
        inp,
        dModel,
        dInner,
        dropout,
        kernelInitializer,
        isTraining: true
    },
    scope = tfex.scope
) {
    return tf.tidy(() => {
        let output = myDense({
            x: args.inp,
            units: args.dInner,
            activation: "relu",
            name: "layer_1",
            scope: scope
                // kernelInitializer: args.kernelInitializer
        }, scope)

        output = tf.layers.dropout({
            rate: args.dropout,
            trainable: args.isTraining
        }).apply(output)

        output = myDense({
            x: output,
            units: args.dInner,
            activation: "relu",
            name: "layer_2"
                // kernelInitializer: args.kernelInitializer
        }, scope)

        output = tf.layers.dropout({
            rate: args.dropout,
            trainable: args.isTraining
        }).apply(output)

        output = tf.add(output, args.inp)

        output = tfex.layers.layerNormalization({ axis: -1 }).apply(output)

        return output
    })
}

export function relShift(
    args = {
        x
    }
) {
    return tf.tidy(() => {
        let x_size = args.x.shape
        let output = tf.pad(args.x, [
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 0]
        ])
        output = tf.reshape(output, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        output = tf.slice(output, [1, 0, 0, 0], [-1, -1, -1, -1])
        output = tf.reshape(output, x_size)
        return output
    })

}

export function relMultiheadAttn(
    args = {
        w,
        r,
        rwBias,
        rrBias,
        attnMask,
        mems,
        dModel,
        nHead,
        dHead,
        dropout,
        dropatt,
        isTraining,
        kernelTnitializer,
    },
    scope = tfex.scope
) {
    return tf.tidy(() => {
        let scale = 1 / (args.dHead ** 0.5)
        let qlen = args.w.shape[0]
        let rlen = args.r.shape[0]
        let bsz = args.w.shape[1]

        let cat = args.mems != null && args.mems.shape.length > 1 ?
            tf.concat([args.mems, args.w], 0) : args.w

        let wHeads = myDense({
                x: cat,
                units: 3 * args.nHead * args.dHead,
                useBias: false,
                // kernelInitializer: args.kernelTnitializer,
                name: 'qkv'
            },
            scope)

        let rHeadK = myDense({
                x: args.r,
                units: args.nHead * args.dHead,
                useBias: false,
                // kernelInitializer: args.kernelTnitializer,
                name: 'r'
            },
            scope)

        let [wHeadQ, wHeadK, wHeadV] = tf.split(wHeads, 3, -1)

        wHeadQ = tf.slice(wHeadQ, wHeadQ.shape[0] - qlen, qlen)

        klen = wHeadK.shape[0]

        wHeadQ = tf.layers.reshape({ targetShape: [qlen, bsz, nHead, dHead] }).apply(wHeadW)
        wHeadK = tf.layers.reshape({ targetShape: [klen, bsz, nHead, dHead] }).apply(wHeadK)
        wHeadV = tf.layers.reshape({ targetShape: [klen, bsz, nHead, dHead] }).apply(wHeadV)

        rHeadK = tf.layers.reshape({ targetShape: [rlen, nHead, dHead] }).apply(rHeadK)

        let rwHeadQ = tf.layers.add().apply([wHeadQ, args.rwBias])
        let rrHeadQ = tf.layers.add().apply([wHeadQ, args.rrBias])

        AC = tfex.einsum('ibnd,jbnd->ijbn', rwHeadQ, wHeadK)

        BD = tfex.einsum('ibnd,jnd->ijbn', rrHeadQ, rHeadK)

        BD = relShift({ x: BD })

        let attnScore = tf.layers.add().apply([AC, BD])
        attnScore = tf.mul(attnScore, tf.tensor([scale]))

        let attnMaskT = attnMask.expandDims(2).expandDims(3)

        attnScore = tf.sub(
            tf.mul(
                attnScore,
                tf.sub(tf.tensor([1]), attnMaskT)
            ),
            tf.mul(
                tf.tensor([1e30]),
                attnMaskT
            )
        )

        let attnProb = tf.layers.softmax({ axis: 1 }).apply(attnScore)
        attnProb = tf.layers.dropout({ rate: args.dropatt, trainable: args.isTraining }).apply(attnProb)

        let attnVec = tf.einsum('ijbn,jbnd->ibnd', attnProb, wHeadV)

        sizeT = attnVec.shape
        attnVec = tf.layers.reshape({ targetShape: [sizeT[0], sizeT[1], args.nHead * args.dHead] }).apply(attnVec)

        let attnOut = myDense({
                x: attnVec,
                units: args.dModel,
                useBias: false,
                // kernelInitializer: args.kernelInitializer,
                name: 'o'
            },
            scope)

        attnOut = tf.layers.dropout({ rate: args.dropout, trainable: args.isTraining }).apply(attnOut)

        output = tfex.layers.layerNormalization({ axis: -1 }).apply(
            tf.layers.add().apply([attnOut, args.w])
        )
        return output
    })
}

export function embeddingLookup(args = { lookupTable, x }) {
    return tf.tidy(() => {
        return tf.gather(args.lookupTable, args.x)
    })
}

export function maskAdaptiveEmbeddingLookup(
    args = {
        x,
        nToken,
        dEmbed,
        dProj,
        cutoffs,
        initializer,
        projInitializer,
        // args.divVal: 1,
        projSameDim: true
    },
    scope = tfex.scope
) {
    return tf.tidy(() => {
        let embScale = args.dProj ** 0.5
            // if (args.divVal == 1) {
        let lookup_table = scope.getVariable('lookup_table', [args.nToken, args.dEmbed], true, "float32")
        let projW
        let y = embeddingLookup({ lookupTable: lookup_table, x: x })
        if (args.dProj != args.dEmbed) {
            projW = scope.getVariable(
                'projW', [args.dEmbed, args.dProj], true, "float32"
            )
            y = tfex.einsum('ibe,ed->ibd', y, projW)
        } else {
            projW = null
        }
        let retParams = [lookup_table, projW]
            // }
        y = tf.mul(y, tf.tensor([embScale]))
        return [y, retParams]
    })
}

export function maskAdaptiveLogsoftmax(
    args = {
        hidden,
        target,
        nToken,
        dEmbed,
        dProj,
        cutoffs,
        params,
        tieProjs,
        initializer: null,
        projInitializer: null,
        divVal: 1,
        projSameDim: true,
        return_mean: true
    },
    scope = tfex.scope
) {
    return tf.tidy(() => {
        let _logit = (x, W, b, proj) => {
            return tf.tidy(() => {
                let y = x
                if (proj != null) {
                    y = tf.einsum('ibd,ed->ibe', y, proj)
                }
                return tf.add(tf.einsum('ibd,nd->ibn', y, W), b)
            })
        }

        let paramsW = params[0]
        let paramsProjs = params[1]


        // if (len(cutoffs) == 0) {}
        let softmax_b = scope.getVariable('bias', [args.nToken],
            // initializer = tf.zeros_initializer()
        )
        let output = _logit(hidden, paramsW, softmax_b, paramsProjs)
        let nll = tf.losses.softmaxCrossEntropy(tf.oneHot(args.target, output.shape[1]), output)
            // }


        if (return_mean) {
            nll = tf.reduce_mean(nll)
        }
        return nll
    })
}

export function _createMask(qlen, mlen, sameLength = false) {
    return tf.tidy(() => {
        let attnMask = tf.ones([qlen, qlen])
        let maskU = tfex.matrixBandPart(attnMask, 0, -1)
        let maskDia = tfex.matrixBandPart(attnMask, 0, 0)
        let attnMaskPad = tf.zeros([qlen, mlen])
        let ret = tf.concat([attnMaskPad, tf.sub(maskU, maskDia)], 1)
            // if (args.sameLength) {
            //     mask_l = tf.matrix_band_part(attnMask, -1, 0)
            //     ret = tf.concat([ret[: , : qlen] + mask_l - maskDia, ret[:, qlen:]], 1)
            // }
        return ret
    })
}


export function _cacheMem(currOut, prevMem, memLen = null) {
    return tf.customGrad((x, y, save) => {
        return tf.tidy(() => {
            // Save x to make sure it's available later for the gradient.
            save([x, y]);
            // Override gradient of our custom x ^ 2 op to be dy * abs(x);
            return {
                value: (() => {
                    let newMem
                    if (args.memLen == null || prevMem == null) {
                        newMem = currOut
                    } else if (args.memLen == 0) {
                        return prevMem
                    } else {
                        newMem = tf.concat([prevMem, currOut], 0)
                        newMem = tf.slice(newMem, [newMem.shape[0] - args.memLen], [newMem.shape[0] - 1])
                    }
                    return newMem
                })(),
                // Note `saved.x` which points to the `x` we saved earlier.
                gradFunc: (dy, saved) => [tf.zeros(saved[0].shape), tf.zeros(saved[1].shape)]
            }
        })
    })(currOut, prevMem)
}

export function transformer(args = {
        decInp: null,
        target: null,
        mems: null,
        nToken: null,
        nLayer: null,
        dModel: null,
        dEmbed: null,
        nHead: null,
        dHead: null,
        dInner: null,
        dropout: null,
        dropatt: null,
        initializer: null,
        isTraining: null,
        projInitializer: null,
        memLen: null,
        cutoffs: [],
        divVal: 1,
        tieProjs: [],
        sameLength: false,
        clampLen: -1,
        inputPerms: null,
        targetPerms: null,
        headTarget: null,
        untieR: false,
        projSameDim: true
    },
    scope = tfex.scope
) {
    // 
    // cutoffs: a list of python int. Cutoffs for adaptive softmax.
    // args.tieProjs: a list of python bools. Whether to tie the projections.
    // use_tpu: if true, use one_hot in embedding lookup and bin-based implementation
    // of adaptive softmax.
    // perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
    // Only used in the adaptive setting.

    let newMems = []
    return tf.tidy(() => {
        let rwBias, rrBias
        if (args.untieR) {
            rwBias = scope.getVariable('rwBias', [args.nLayer, args.nHead, args.dHead],
                initializer = args.initializer)
            rrBias = scope.getVariable('rrBias', [args.nLayer, args.nHead, args.dHead],
                initializer = args.initializer)
        } else {
            rwBias = scope.getVariable('rwBias', [args.nHead, args.dHead],
                initializer = args.initializer)
            rrBias = scope.getVariable('rrBias', [args.nHead, args.dHead],
                initializer = args.initializer)
        }

        let qlen = tf.shape(args.decInp)[0]
        let mlen = args.mems != null ? args.mems[0].shape[0] : 0
        let klen = mlen + qlen

        if (args.projInitializer == null) {
            args.projInitializer = args.initializer
        }
        let lookupFn = maskAdaptiveEmbeddingLookup

        let [embeddings, sharedParams] = lookupFn(
            x = args.decInp,
            args.nToken = args.nToken,
            args.dEmbed = args.dEmbed,
            dProj = args.dModel,
            cutoffs = args.cutoffs,
            initializer = args.initializer,
            args.projInitializer = args.projInitializer,
            args.divVal = args.divVal,
            perms = args.inputPerms,
            args.projSameDim = args.projSameDim)

        let attnMask = _createMask(qlen, mlen, args.sameLength)

        let posSeq = tf.range(klen - 1, -1, -1.0)
        if (args.clampLen > 0) {
            posSeq = tf.minimum(posSeq, args.clampLen)
        }
        let invFreq = 1 / (10000 ** (tf.range(0, args.dModel, 2.0) / args.dModel))
        let posEmb = positionalEmbedding(posSeq, invFreq)

        let output = tf.layers.dropout(embeddings, args.dropout, training = args.isTraining)
        posEmb = tf.layers.dropout(posEmb, args.dropout, training = args.isTraining)

        if (args.mems == null) {
            mems = new Array(args.nLayer).fill(null)
        }

        for (i in range(args.nLayer)) { //cache new mems
            newMems.push(_cacheMem(output, mems[i], args.memLen))
            output = tf.tidy(() => {
                let layerScope = scope.variableScope(`layer_${i}`)
                output = relMultiheadAttn({
                        w: output,
                        r: posEmb,
                        rwBias: !args.untieR ? rwBias : rwBias[i],
                        rrBias: !args.untieR ? rrBias : rrBias[i],
                        attnMask: attnMask,
                        mems: mems[i],
                        dModel: args.dModel,
                        nHead: args.nHead,
                        dHead: args.dHead,
                        dropout: args.dropout,
                        dropatt: args.dropatt,
                        isTraining: args.isTraining,
                        kernel_initializer: args.initializer
                    },
                    layerScope.variableScope("rel_attn")
                )
                output = positionwiseFF({
                        inp: output,
                        dModel: args.dModel,
                        adInner: args.dInner,
                        dropout: args.dropout,
                        kernel_initializer: args.initializer,
                        isTraining: args.isTraining
                    },
                    layerScope.variableScope("ff")
                )
                return output
            })
        }
        output = tf.layers.dropout(output, args.dropout, training = args.isTraining)

        let logsoftmax_fn = maskAdaptiveLogsoftmax
        let loss = logsoftmax_fn({
                hidden: output,
                target: args.target,
                nToken: args.nToken,
                dEmbed: args.dEmbed,
                dProj: args.dModel,
                cutoffs: args.cutoffs,
                params: sharedParams,
                tieProjs: args.tieProjs,
                initializer: args.initializer,
                projInitializer: args.projInitializer,
                divVal: args.divVal,
                perms: args.targetPerms,
                headTarget: args.headTarget,
                projSameDim: args.projSameDim
            },
            scope.variableScope("adaptive_softmax")
        )
        return [loss, newMems]
    })
}