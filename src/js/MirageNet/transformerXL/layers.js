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
        isTraining = true
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
        d_proj,
        cutoffs,
        initializer,
        projInitializer,
        // args.divVal: 1,
        projSameDim: true
    },
    scope = tfex.scope
) {
    return tf.tidy(() => {
        emb_scale = d_proj ** 0.5
            // if (args.divVal == 1) {
        let lookup_table = scope.getVariable('lookup_table', [args.nToken, args.dEmbed], true, "float32")
        let proj_W
        let y = embeddingLookup({ lookupTable: lookup_table, x: x })
        if (d_proj != args.dEmbed) {
            proj_W = scope.getVariable(
                'proj_W', [args.dEmbed, d_proj], true, "float32"
            )
            y = tfex.einsum('ibe,ed->ibd', y, proj_W)
        } else {
            proj_W = null
        }
        let ret_params = [lookup_table, proj_W]
            // }
        y = tf.mul(y, tf.tensor([emb_scale]))
        return [y, ret_params]
    })
}

export function maskAdaptiveLogsoftmax(
    args = {
        hidden,
        target,
        nToken,
        dEmbed,
        d_proj,
        cutoffs,
        params,
        tieProjs,
        initializer = null,
        projInitializer = null,
        divVal = 1,
        projSameDim = true,
        return_mean = true
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

        params_W = params[0]
        params_projs = params[1]


        // if (len(cutoffs) == 0) {}
        let softmax_b = scope.getVariable('bias', [args.nToken],
            // initializer = tf.zeros_initializer()
        )
        let output = _logit(hidden, params_W, softmax_b, params_projs)
        let nll = tf.losses.softmaxCrossEntropy(tf.oneHot(args.target, output.shape[1]), output)
            // }


        if (return_mean) {
            nll = tf.reduce_mean(nll)
        }
        return nll
    })
}

export function _create_mask(qlen, mlen, sameLength = false) {
    return tf.tidy(() => {
        attn_mask = tf.ones([qlen, qlen])
        mask_u = tfex.matrixBandPart(attn_mask, 0, -1)
        mask_dia = tfex.matrixBandPart(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen])
        ret = tf.concat([attn_mask_pad, tf.sub(mask_u, mask_dia)], 1)
            // if (args.sameLength) {
            //     mask_l = tf.matrix_band_part(attn_mask, -1, 0)
            //     ret = tf.concat([ret[: , : qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
            // }
        return ret
    })
}


export function _cache_mem(curr_out, prev_mem, memLen = null) {
    return tf.customGrad((x, y, save) => {
        return tf.tidy(() => {
            // Save x to make sure it's available later for the gradient.
            save([x, y]);
            // Override gradient of our custom x ^ 2 op to be dy * abs(x);
            return {
                value: (() => {
                    let new_mem
                    if (args.memLen == null || prev_mem == null) {
                        new_mem = curr_out
                    } else if (args.memLen == 0) {
                        return prev_mem
                    } else {
                        new_mem = tf.concat([prev_mem, curr_out], 0)
                        new_mem = tf.slice(new_mem, [new_mem.shape[0] - args.memLen], [new_mem.shape[0] - 1])
                    }
                    return new_mem
                })(),
                // Note `saved.x` which points to the `x` we saved earlier.
                gradFunc: (dy, saved) => [tf.zeros(saved[0].shape), tf.zeros(saved[1].shape)]
            }
        })
    })(curr_out, prev_mem)
}

export function transformer(args = {
        decInp,
        target,
        mems,
        nToken,
        nLayer,
        dModel,
        dEmbed,
        nHead,
        dHead,
        dInner,
        dropout,
        dropatt,
        initializer,
        isTraining,
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

    let new_mems = []
    return tf.tidy(() => {
        let rwBias, rrBias
        if (args.untieR) {
            rwBias = scope.getVariable('rwBias', [args.nLayer, args.nHead, args.dHead],
                initializer = initializer)
            rrBias = scope.getVariable('rrBias', [args.nLayer, args.nHead, args.dHead],
                initializer = initializer)
        } else {
            rwBias = scope.getVariable('rwBias', [args.nHead, args.dHead],
                initializer = initializer)
            rrBias = scope.getVariable('rrBias', [args.nHead, args.dHead],
                initializer = initializer)
        }

        let qlen = tf.shape(args.decInp)[0]
        let mlen = mems != null ? tf.shape(mems[0])[0] : 0
        let klen = mlen + qlen

        if (args.projInitializer == null) {
            args.projInitializer = initializer
        }
        let lookupFn = maskAdaptiveEmbeddingLookup

        let [embeddings, shared_params] = lookupFn(
            x = args.decInp,
            args.nToken = args.nToken,
            args.dEmbed = args.dEmbed,
            d_proj = args.dModel,
            cutoffs = cutoffs,
            initializer = initializer,
            args.projInitializer = args.projInitializer,
            args.divVal = args.divVal,
            perms = args.inputPerms,
            args.projSameDim = args.projSameDim)

        attn_mask = _create_mask(qlen, mlen, args.sameLength)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if (args.clampLen > 0) {
            pos_seq = tf.minimum(pos_seq, args.clampLen)
        }
        inv_freq = 1 / (10000 ** (tf.range(0, args.dModel, 2.0) / args.dModel))
        pos_emb = positionalEmbedding(pos_seq, inv_freq)

        let output = tf.layers.dropout(embeddings, dropout, training = args.isTraining)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training = args.isTraining)

        if (mems == null) {
            mems = [null] * args.nLayer
        }

        for (i in range(args.nLayer)) { //cache new mems
            new_mems.push(_cache_mem(output, mems[i], args.memLen))
            output = tf.tidy(() => {
                let layerScope = scope.variableScope(`layer_${i}`)
                output = relMultiheadAttn({
                        w = output,
                        r = pos_emb,
                        rwBias = !args.untieR ? rwBias : rwBias[i],
                        rrBias = !args.untieR ? rrBias : rrBias[i],
                        attn_mask = attn_mask,
                        mems = mems[i],
                        dModel = args.dModel,
                        nHead = args.nHead,
                        dHead = args.dHead,
                        dropout = dropout,
                        dropatt = dropatt,
                        isTraining = args.isTraining,
                        kernel_initializer = initializer
                    },
                    layerScope.variableScope("rel_attn")
                )
                output = positionwiseFF({
                        inp = output,
                        dModel = args.dModel,
                        adInner = args.dInner,
                        dropout = dropout,
                        kernel_initializer = initializer,
                        isTraining = args.isTraining
                    },
                    layerScope.variableScope("ff")
                )
                return output
            })
        }
        output = tf.layers.dropout(output, dropout, training = args.isTraining)

        logsoftmax_fn = maskAdaptiveLogsoftmax
        loss = logsoftmax_fn({
            hidden = output,
            target = target,
            nToken = args.nToken,
            dEmbed = args.dEmbed,
            d_proj = args.dModel,
            cutoffs = cutoffs,
            params = shared_params,
            tieProjs = args.tieProjs,
            initializer = initializer,
            projInitializer = args.projInitializer,
            divVal = args.divVal,
            perms = args.targetPerms,
            headTarget = args.headTarget,
            projSameDim = args.projSameDim,
            scope: scope.variableScope("adaptive_softmax")
        })
        return [loss, new_mems]
    })
}