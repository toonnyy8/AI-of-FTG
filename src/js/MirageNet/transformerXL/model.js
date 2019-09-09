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

        return tfex.tool.tensorPtr(tf.reshape(args.x, [-1, kernelShape[0]]))
            .sequence(
                tptr => {
                    tptr.assign(
                        tf.dot(
                            tptr.read(),
                            scope.getVariable(`${args.name}_kernel`, kernelShape, "float32", args.kernelInitializer, true)
                        )
                    )
                    tptr.assign(
                        tf.reshape(tptr.read(), outputShape)
                    )
                    if (args.useBias) {
                        tptr.assign(
                            tf.add(
                                tptr.read(),
                                scope.getVariable(`${args.name}_bias`, [args.units], "float32", args.kernelInitializer, true)
                            )
                        )
                    }
                    if (args.activation != null) {
                        tptr.assign(
                            tf[args.activation](tptr.read())
                        )
                    }
                }
            )
            .read()
    })
}

export function positionalEmbedding(
    args = {
        bsz: null,
        posSeq,
        invFreq
    }
) {
    //console.log("positionalEmbedding")
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
    //console.log("positionwiseFF")
    return tf.tidy(() => {
        let output = myDense({
            x: args.inp,
            units: args.dInner,
            activation: "relu",
            name: "layer_1",
            scope: scope
            // kernelInitializer: args.kernelInitializer
        }, scope)

        output = tf.dropout(output, args.dropout)

        output = myDense({
            x: output,
            units: args.dModel,
            activation: "relu",
            name: "layer_2"
            // kernelInitializer: args.kernelInitializer
        }, scope)

        output = tf.dropout(output, args.dropout)

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
    //console.log("relMultiheadAttn")
    return tf.tidy(() => {
        //console.log(tf.memory())
        let scale = 1 / (args.dHead ** 0.5)
        let qlen = args.w.shape[0]
        let rlen = args.r.shape[0]
        let bsz = args.w.shape[1]

        let cat = tfex.tool.tensorPtr(args.mem != null && args.mem.shape.length > 1 ?
            tf.concat([args.mem, args.w], 0) : args.w)

        let wHeads = tfex.tool.tensorPtr(myDense({
            x: cat.read(),
            units: 3 * args.nHead * args.dHead,
            useBias: false,
            kernelInitializer: args.kernelTnitializer,
            name: 'qkv'
        },
            scope))

        let rHeadK = tfex.tool.tensorPtr(myDense({
            x: args.r,
            units: args.nHead * args.dHead,
            useBias: false,
            kernelInitializer: args.kernelTnitializer,
            name: 'r'
        },
            scope))

        let [whq, whk, whv] = tf.split(wHeads.read(), 3, -1)
        wHeads.assign(null)
        let wHeadQ = tfex.tool.tensorPtr(whq)
        let wHeadK = tfex.tool.tensorPtr(whk)
        let wHeadV = tfex.tool.tensorPtr(whv)
        wHeadQ.assign(tf.slice(wHeadQ.read(), wHeadQ.read().shape[0] - qlen, qlen))

        let klen = wHeadK.read().shape[0]

        wHeadQ.assign(tf.reshape(wHeadQ.read(), [qlen, bsz, args.nHead, args.dHead]))
        wHeadK.assign(tf.reshape(wHeadK.read(), [klen, bsz, args.nHead, args.dHead]))
        wHeadV.assign(tf.reshape(wHeadV.read(), [klen, bsz, args.nHead, args.dHead]))

        rHeadK.assign(tf.reshape(rHeadK.read(), [rlen, args.nHead, args.dHead]))

        let rwHeadQ = tfex.tool.tensorPtr(tf.add(wHeadQ.read(), args.rwBias))
        let rrHeadQ = tfex.tool.tensorPtr(tf.add(wHeadQ.read(), args.rrBias))
        wHeadQ.assign(null)
        //console.log(tf.memory())
        let AC = tfex.tool.tensorPtr(tfex.einsum('ibnd,jbnd->ijbn', rwHeadQ.read(), wHeadK.read()))
        wHeadK.assign(null)
        //console.log(tf.memory())
        let BD = tfex.tool.tensorPtr(tfex.einsum('ibnd,jnd->ijbn', rrHeadQ.read(), rHeadK.read()))
        rHeadK.assign(null)
        //console.log(tf.memory())
        BD.assign(relShift({ x: BD.read() }))
        rwHeadQ.assign(null)
        rrHeadQ.assign(null)

        let attnScore = tfex.tool.tensorPtr(tf.add(AC.read(), BD.read()))
        AC.assign(null)
        BD.assign(null)

        attnScore.assign(tf.mul(attnScore.read(), tf.tensor([scale])))

        let attnMaskT = tfex.tool.tensorPtr(args.attnMask.expandDims(2).expandDims(3))

        attnScore.assign(
            tf.sub(
                tf.mul(
                    attnScore.read(),
                    tf.sub(tf.tensor([1]), attnMaskT.read())
                ),
                tf.mul(
                    tf.tensor([1e30]),
                    attnMaskT.read()
                )
            )
        )

        let attnProb = tfex.tool.tensorPtr(tf.tidy(() => {
            return tf.div(tf.exp(attnScore.read()), tf.sum(tf.exp(attnScore.read()), 1, true))
        }))
        // tf.softmax(attnScore, 1)
        attnProb.assign(tf.dropout(attnProb.read(), args.dropatt))
        //console.log(tf.memory())
        let attnVec = tfex.tool.tensorPtr(tfex.einsum('ijbn,jbnd->ibnd', attnProb.read(), wHeadV.read()))
        wHeadV.assign(null)
        //console.log(tf.memory())

        let sizeT = attnVec.read().shape
        attnVec.assign(tf.reshape(attnVec.read(), [sizeT[0], sizeT[1], args.nHead * args.dHead]))

        let attnOut = tfex.tool.tensorPtr(
            myDense({
                x: attnVec.read(),
                units: args.dModel,
                useBias: false,
                kernelInitializer: args.kernelInitializer,
                name: 'o'
            },
                scope)
        )

        attnOut.assign(tf.dropout(attnOut.read(), args.dropout))

        let output = tfex.tool.tensorPtr(tfex.layers.layerNormalization({ axis: -1 }).apply(
            tf.add(attnOut.read(), args.w)
        ))
        return output.read()
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
    //console.log("maskAdaptiveEmbeddingLookup")
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
    //console.log("maskAdaptiveLogsoftmax")
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
            newMem = tf.concat([prevMem, currOut], 0)
            newMem = tf.slice(newMem, [newMem.shape[0] - memLen], [memLen])
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

        let attnMask = _createMask(qlen, mlen, args.sameLength)

        let posSeq = tfex.tool.tensorPtr(tf.range(klen - 1, -1, -1.0))
            .sequence(tptr => {
                if (args.clampLen > 0) {
                    tptr.assign(tf.minimum(tptr.read(), args.clampLen))
                }
            })

        let invFreq = tfex.tool.tensorPtr(tf.range(0, args.dModel, 2.0))
            .sequence(tptr => tptr.assign(tf.div(tptr.read(), args.dModel)))
            .sequence(tptr => tptr.assign(tf.pow(10000, tptr.read())))
            .sequence(tptr => tptr.assign(tf.div(1, tptr.read())))

        let posEmb = tfex.tool.tensorPtr(positionalEmbedding({ posSeq: posSeq.read(), invFreq: invFreq.read() }))

        let output = tfex.tool.tensorPtr(tf.dropout(embeddings, args.dropout))
        posEmb.assign(tf.dropout(posEmb.read(), args.dropout))

        if (mems == null) {
            mems = new Array(args.nLayer).fill(null)
        }

        for (let i = 0; i < args.nLayer; i++) { //cache new mems
            newMems.push(_cacheMem(output.read(), mems[i], args.memLen))
            let layerScope = scope.variableScope(`layer_${i}`)
            output.sequence(tptr => {
                tptr.assign(
                    relMultiheadAttn({
                        w: tptr.read(),
                        r: posEmb.read(),
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
                )
            }).sequence(tptr => {
                tptr.assign(
                    positionwiseFF({
                        inp: tptr.read(),
                        dModel: args.dModel,
                        dInner: args.dInner,
                        dropout: args.dropout,
                        kernelInitializer: args.initializer,
                        isTraining: args.isTraining
                    },
                        layerScope.variableScope("ff")
                    )
                )
            })
        }
        output.assign(tf.dropout(output.read(), args.dropout))


        let logsoftmax_fn = maskAdaptiveLogsoftmax
        let loss = logsoftmax_fn({
            hidden: output.read(),
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

        return [loss, tf.stack(newMems), output.read()]
    })
}