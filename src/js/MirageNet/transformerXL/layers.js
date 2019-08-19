import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../../lib/tfjs-extensions/src"

//----------positional embedding
export function positionalEmbedding(
    args = {
        bsz: null,
        posSeq,
        invFreq
    }
) {
    return tfex.layers.lambda({
        func: (posSeq, invFreq) => {
            let sinusoidInp = tfex.einsum('i,j->ij', posSeq, invFreq)
            let posEmb = tf.concat([tf.sin(sinusoidInp), tf.cos(sinusoidInp)], -1)
            if (args.bsz != null) {
                return posEmb.expandDims(1).tile([1, args.bsz, 1])
            }
            else {
                return posEmb.expandDims(1)
            }
        }
    }).apply([args.posSeq, args.invFreq])
}
//----------positionwise feed forward
export function positionwise_FF(
    args = {
        inp,
        dModel,
        dInner,
        dropout,
        kernelInitializer,
        isTraining = true
    }
) {
    output = args.inp
    let output = tf.layers.dense({
        units: args.dInner,
        activation: "relu",
        kernelInitializer: args.kernelInitializer
    }).apply(args.inp)
    output = tf.layers.dropout({
        rate: args.dropout,
        trainable: args.isTraining
    }).apply(output)

    output = tf.layers.dense({
        units: args.dModel,
        activation: "relu",
        kernelInitializer: args.kernelInitializer
    }).apply(output)
    output = tf.layers.dropout({
        rate: args.dropout,
        trainable: args.isTraining
    }).apply(output)

    output = tfex.layers.lambda({
        func: (x, y) => {
            return tf.add(x, y)
        }
    }).apply([output, args.inp])
    output = tfex.layers.layerNormalization({ axis: -1 }).apply(output)

    return output
}
//----------relative shift
export function relShift(
    args = {
        x
    }
) {
    return tfex.layers.lambda({
        func: (x) => {
            let x_size = x.shape
            x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
            x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
            x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
            x = tf.reshape(x, x_size)
            return x
        }
    }).apply(args.x)
}
//----------relative multihead attnention
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
        kernelTnitializer
    }
) {
    let scale = 1 / (args.dHead ** 0.5)
    let qlen = args.w.shape[0]
    let rlen = args.r.shape[0]
    let bsz = args.w.shape[1]

    let cat = args.mems != null && args.mems.shape.length > 1 ?
        tfex.layers.lambda({
            func: (mems, w) => {
                return tf.concat([mems, w], 0)
            }
        }).apply([args.mems, args.w]) :
        tfex.layers.lambda({
            func: (w) => {
                return w
            }
        }).apply(args.w)

    let wHeads = tf.layers.dense(
        {
            units: 3 * args.nHead * args.dHead,
            useBias: false,
            kernelInitializer: args.kernelTnitializer,
            name: 'qkv'
        }
    ).apply(cat)
    let rHeadK = tf.layers.dense({
        units: args.nHead * args.dHead,
        useBias: false,
        kernelInitializer: args.kernelTnitializer,
        name: 'r'
    }
    ).apply(args.r)

    let [wHeadQ, wHeadK, wHeadV] = tfex.layers.lambda({
        func: (x) => {
            return tf.split(x, 3, -1)
        }
    }).apply(wHeads)

    wHeadQ = tfex.layers.lambda({
        func: (x) => {
            return tf.slice(x, x.shape[0] - qlen, qlen)
        }
    }).apply(wHeadQ)

    klen = wHeadK.shape[0]

    wHeadQ = tf.layers.reshape({ targetShape: [qlen, bsz, nHead, dHead] }).apply(wHeadW)
    wHeadK = tf.layers.reshape({ targetShape: [klen, bsz, nHead, dHead] }).apply(wHeadK)
    wHeadV = tf.layers.reshape({ targetShape: [klen, bsz, nHead, dHead] }).apply(wHeadV)

    rHeadK = tf.layers.reshape({ targetShape: [rlen, nHead, dHead] }).apply(rHeadK)

    let rwHeadQ = tf.layers.add().apply([wHeadQ, args.rwBias])
    let rrHeadQ = tf.layers.add().apply([wHeadQ, args.rrBias])

    AC = tfex.layers.lambda({
        func: (x, y) => {
            return tfex.einsum('ibnd,jbnd->ijbn', x, y)
        }
    }).apply([rwHeadQ, wHeadK])

    BD = tfex.layers.lambda({
        func: (x, y) => {
            return tfex.einsum('ibnd,jnd->ijbn', x, y)
        }
    }).apply([rrHeadQ, rHeadK])
    BD = relShift({ x: BD })

    let attnScore = tf.layers.add().apply([AC, BD])
    attnScore = tfex.layers.lambda({
        func: (x) => {
            return tf.mul(x, tf.tensor([scale]))
        }
    }).apply(attnScore)

    let attnMaskT = tfex.layers.lambda({
        func: (x) => {
            return x.expandDims(2).expandDims(3)
        }
    }).apply(attnMask)

    attnScore = tf.layers.add().apply(
        [
            tf.layers.multiply().apply(
                [
                    attnScore,
                    tfex.layers.lambda({
                        func: (x) => {
                            return tf.sub(tf.tensor([1]), x)
                        }
                    }).apply(attnMaskT)
                ]
            ),
            tfex.layers.lambda({
                func: (x) => {
                    return tf.mul(tf.tensor([- 1e30]), x)
                }
            }).apply(attnMaskT)
        ]
    )

    let attnProb = tf.layers.softmax({ axis: 1 }).apply(attnScore)
    attnProb = tf.layers.dropout({ rate: args.dropatt, trainable: args.isTraining }).apply(attnProb)

    let attnVec = tfex.layers.lambda({
        func: (x, y) => {
            return tf.einsum('ijbn,jbnd->ibnd', x, y)
        }
    }).apply([attnProb, wHeadV])
    sizeT = attnVec.shape
    attnVec = tf.layers.reshape({ targetShape: [sizeT[0], sizeT[1], args.nHead * args.dHead] }).apply(attnVec)

    let attnOut = tf.layers.dense({
        units: args.dModel,
        useBias: false,
        kernelInitializer: args.kernelInitializer,
        name: 'o'
    }).apply(attnVec)
    attnOut = tf.layers.dropout({ rate: args.dropout, trainable: args.isTraining }).apply(attnOut)

    output = tfex.layers.layerNormalization({ axis: -1 }).apply(
        tf.layers.add().apply([attnOut, args.w])
    )
    return output
}
//----------embedding lookup
export function embeddingLookup(args = { lookupTable, x }) {
    return tfex.layers.lambda({
        func: (lookupTable, x) => {
            return tf.gather(lookupTable, x)
        }
    }).apply([args.lookupTable, args.x])
}

