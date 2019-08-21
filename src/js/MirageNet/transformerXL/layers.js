import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"

function myDense(
    args = {
        scope: tfex.scope,
        x,
        units,
        activation,
        kernelInitializer,
        name: "",
        useBias: true
    }
) {
    return tf.tidy(() => {
        let output = tf.dot(x,
            args.scope.getVariable(`${name}_kernel`, x.concat([units]), true, "float32"))

        if (useBias) {
            output = tf.add(output,
                args.scope.getVariable(`${name}_bias`, x.concat([units]), true, "float32")
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

export function positionwise_FF(
    args = {
        inp,
        dModel,
        dInner,
        dropout,
        kernelInitializer,
        isTraining = true,
        scope: tfex.scope
    }
) {
    return tf.tidy(() => {
        let output = myDense({
            x: args.inp,
            units: args.dInner,
            activation: "relu",
            name: "layer_1",
            scope: args.scope
                // kernelInitializer: args.kernelInitializer
        })

        output = tf.layers.dropout({
            rate: args.dropout,
            trainable: args.isTraining
        }).apply(output)

        output = myDense({
            x: output,
            units: args.dInner,
            activation: "relu",
            name: "layer_2",
            scope: args.scope
                // kernelInitializer: args.kernelInitializer
        })

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
        scope: tfex.scope
    }
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
            name: 'qkv',
            scope: args.scope
        })

        let rHeadK = myDense({
            x: args.r,
            units: args.nHead * args.dHead,
            useBias: false,
            // kernelInitializer: args.kernelTnitializer,
            name: 'r',
            scope: args.scope
        })

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
            name: 'o',
            scope: args.scope
        })

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
        n_token,
        d_embed,
        d_proj,
        cutoffs,
        initializer,
        proj_initializer,
        // div_val: 1,
        proj_same_dim: true,
        scope: tfex.scope
    }
) {
    return tf.tidy(() => {
        emb_scale = d_proj ** 0.5
            // if (div_val == 1) {
        let lookup_table = args.scope.getVariable('lookup_table', [n_token, d_embed], true, "float32")
        let proj_W
        let y = embeddingLookup({ lookupTable: lookup_table, x: x })
        if (d_proj != d_embed) {
            proj_W = args.scope.getVariable(
                'proj_W', [d_embed, d_proj], true, "float32"
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