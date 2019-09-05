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
        let kernelShape = [args.x.shape[args.x.shape.length - 1], args.units]
        let outputShape = args.x.shape.slice()
        outputShape.pop()
        outputShape.push(args.units)

        let output = tf.dot(tf.reshape(args.x, [-1, kernelShape[0]]),
            scope.getVariable(`${args.name}_kernel`, kernelShape, "float32", args.kernelInitializer, true)
        )
        output = tf.reshape(output, outputShape)
        if (args.useBias) {
            output = tf.add(output,
                scope.getVariable(`${args.name}_bias`, [args.units], "float32", args.kernelInitializer, true)
            )
        }

        if (args.activation != null) {
            output = tf[args.activation](output)
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
    console.log("positionalEmbedding")
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
    console.log("positionwiseFF")
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
            units: args.dModel,
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
        mem,
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
    console.log("relMultiheadAttn")
    return tf.tidy(() => {
        let scale = 1 / (args.dHead ** 0.5)
        let qlen = args.w.shape[0]
        let rlen = args.r.shape[0]
        let bsz = args.w.shape[1]

        let cat = args.mem != null && args.mem.shape.length > 1 ?
            tf.concat([args.mem, args.w], 0) : args.w

        let wHeads = myDense({
            x: cat,
            units: 3 * args.nHead * args.dHead,
            useBias: false,
            kernelInitializer: args.kernelTnitializer,
            name: 'qkv'
        },
            scope)

        let rHeadK = myDense({
            x: args.r,
            units: args.nHead * args.dHead,
            useBias: false,
            kernelInitializer: args.kernelTnitializer,
            name: 'r'
        },
            scope)

        let [wHeadQ, wHeadK, wHeadV] = tf.split(wHeads, 3, -1)
        wHeadQ = tf.slice(wHeadQ, wHeadQ.shape[0] - qlen, qlen)

        let klen = wHeadK.shape[0]

        wHeadQ = tf.reshape(wHeadQ, [qlen, bsz, args.nHead, args.dHead])
        wHeadK = tf.reshape(wHeadK, [klen, bsz, args.nHead, args.dHead])
        wHeadV = tf.reshape(wHeadV, [klen, bsz, args.nHead, args.dHead])

        rHeadK = tf.reshape(rHeadK, [rlen, args.nHead, args.dHead])

        let rwHeadQ = tf.add(wHeadQ, args.rwBias)
        let rrHeadQ = tf.add(wHeadQ, args.rrBias)

        let AC = tfex.einsum('ibnd,jbnd->ijbn', rwHeadQ, wHeadK)

        let BD = tfex.einsum('ibnd,jnd->ijbn', rrHeadQ, rHeadK)

        BD = relShift({ x: BD })

        let attnScore = tf.add(AC, BD)
        attnScore = tf.mul(attnScore, tf.tensor([scale]))

        let attnMaskT = args.attnMask.expandDims(2).expandDims(3)

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

        let attnProb = tf.tidy(() => {
            return tf.div(tf.exp(attnScore), tf.sum(tf.exp(attnScore), 1, true))
        })
        // tf.softmax(attnScore, 1)
        attnProb = tf.layers.dropout({ rate: args.dropatt, trainable: args.isTraining }).apply(attnProb)

        let attnVec = tfex.einsum('ijbn,jbnd->ibnd', attnProb, wHeadV)

        let sizeT = attnVec.shape
        attnVec = tf.reshape(attnVec, [sizeT[0], sizeT[1], args.nHead * args.dHead])

        let attnOut = myDense({
            x: attnVec,
            units: args.dModel,
            useBias: false,
            kernelInitializer: args.kernelInitializer,
            name: 'o'
        },
            scope)

        attnOut = tf.dropout(attnOut, args.dropout)

        let output = tfex.layers.layerNormalization({ axis: -1 }).apply(
            tf.add(attnOut, args.w)
        )
        return output
    })
}

export function embeddingLookup(args = { lookupTable, x }) {
    return tf.tidy(() => {
        return tf.gather(args.lookupTable, tf.cast(args.x, "int32"))
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
    console.log("maskAdaptiveEmbeddingLookup")
    return tf.tidy(() => {
        let embScale = args.dProj ** 0.5
        // if (args.divVal == 1) {
        let lookupTable = scope.getVariable('lookupTable', [args.nToken, args.dEmbed], "float32", args.initializer, true)
        let projW
        let y = embeddingLookup({ lookupTable: lookupTable, x: args.x })
        if (args.dProj != args.dEmbed) {
            projW = scope.getVariable(
                'projW', [args.dEmbed, args.dProj], "float32", args.initializer, true
            )
            y = tfex.einsum('ibe,ed->ibd', y, projW)
        } else {
            projW = null
        }
        let retParams = [lookupTable, projW]
        // }
        y = tf.mul(y, tf.tensor([embScale]))

        return [y, retParams]
    })
}

export async function maskAdaptiveLogsoftmax(
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
    console.log("maskAdaptiveLogsoftmax")
    return tf.tidy(() => {
        let _logit = (x, W, b, proj) => {
            return tf.tidy(() => {
                let y = x
                if (proj != null) {
                    y = tfex.einsum('ibd,ed->ibe', y, proj)
                }
                return tf.add(tfex.einsum('ibd,nd->ibn', y, W), b)
            })
        }

        let paramsW = args.params[0]
        let paramsProjs = args.params[1]

        // if (len(cutoffs) == 0) {}
        let softmax_b = scope.getVariable('bias', [args.nToken], "float32", tf.initializers.zeros(), true)
        let output = _logit(args.hidden, paramsW, softmax_b, paramsProjs)

        let nll = tf.losses.softmaxCrossEntropy(tf.oneHot(tf.cast(args.target, "int32"), output.shape[2]), output)
        // }


        if (args.return_mean) {
            nll = tf.mean(nll)
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
    return tf.tidy(() => {
        let newMem
        if (memLen == null || prevMem == null) {
            newMem = currOut
        } else if (memLen == 0) {
            return prevMem
        } else {
            console.log(newMem)
            newMem = tf.concat([prevMem, currOut], 0)
            newMem = tf.slice(newMem, [newMem.shape[0] - memLen], [memLen])
            console.log(newMem)
        }
        return tfex.stopGradient(newMem)
    })
}

export function transformer(args = {
    decInp: null,
    target: null,
    mems: null, //stack tensor
    nToken: null, //字典大小，在此視為一次傳入的狀態數(??)
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

    return tf.tidy(() => {
        let newMems = []
        let mems = args.mems != null ? tf.unstack(args.mems) : null
        let rwBias, rrBias
        if (args.untieR) {
            rwBias = scope.getVariable('rwBias', [args.nLayer, args.nHead, args.dHead], "float32", args.initializer)
            rrBias = scope.getVariable('rrBias', [args.nLayer, args.nHead, args.dHead], "float32", args.initializer)
        } else {
            rwBias = scope.getVariable('rwBias', [args.nHead, args.dHead], "float32", args.initializer)
            rrBias = scope.getVariable('rrBias', [args.nHead, args.dHead], "float32", args.initializer)
        }

        let qlen = args.decInp.shape[0]
        let mlen = mems != null ? mems[0].shape[0] : 0
        let klen = mlen + qlen

        if (args.projInitializer == null) {
            args.projInitializer = args.initializer
        }
        let lookupFn = maskAdaptiveEmbeddingLookup
        console.log(tf.memory())

        let [embeddings, sharedParams] = lookupFn({
            x: args.decInp,
            nToken: args.nToken,
            dEmbed: args.dEmbed,
            dProj: args.dModel,
            cutoffs: args.cutoffs,
            initializer: args.initializer,
            projInitializer: args.projInitializer,
            divVal: args.divVal,
            perms: args.inputPerms,
            projSameDim: args.projSameDim
        },
            scope.variableScope("adaptiveEmbed")
        )

        console.log(tf.memory())
        let attnMask = _createMask(qlen, mlen, args.sameLength)

        let posSeq = tf.range(klen - 1, -1, -1.0)
        if (args.clampLen > 0) {
            posSeq = tf.minimum(posSeq, args.clampLen)
        }
        let invFreq = tf.div(1, tf.pow(10000, (tf.div(tf.range(0, args.dModel, 2.0), args.dModel))))

        console.log(tf.memory())
        let posEmb = positionalEmbedding({ posSeq: posSeq, invFreq: invFreq })

        let output = tf.dropout(embeddings, args.dropout)
        posEmb = tf.dropout(posEmb, args.dropout)

        if (mems == null) {
            mems = new Array(args.nLayer).fill(null)
        }

        for (let i = 0; i < args.nLayer; i++) { //cache new mems
            console.log(tf.memory())
            newMems.push(_cacheMem(output, mems[i], args.memLen))
            output = tf.tidy(() => {
                let layerScope = scope.variableScope(`layer_${i}`)
                output = relMultiheadAttn({
                    w: output,
                    r: posEmb,
                    rwBias: !args.untieR ? rwBias : rwBias[i],
                    rrBias: !args.untieR ? rrBias : rrBias[i],
                    attnMask: attnMask,
                    mem: mems[i],
                    dModel: args.dModel,
                    nHead: args.nHead,
                    dHead: args.dHead,
                    dropout: args.dropout,
                    dropatt: args.dropatt,
                    isTraining: args.isTraining,
                    kernelInitializer: args.initializer
                },
                    layerScope.variableScope("rel_attn")
                )
                output = positionwiseFF({
                    inp: output,
                    dModel: args.dModel,
                    dInner: args.dInner,
                    dropout: args.dropout,
                    kernelInitializer: args.initializer,
                    isTraining: args.isTraining
                },
                    layerScope.variableScope("ff")
                )
                return output
            })
        }
        output = tf.dropout(output, args.dropout)

        let logsoftmax_fn = maskAdaptiveLogsoftmax
        console.log(tf.memory())
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
        return [loss, tf.stack(newMems), output]
    })
}