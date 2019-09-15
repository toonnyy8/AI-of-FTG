import * as tf from "@tensorflow/tfjs"
import * as tool from "./tool"

export function repeatElements(x, rep, axis) {
    return tf.tidy(() => {
        let reps = [...Array(x.shape.length)].map(_ => 1)
        reps[axis] = rep
        return tf.tile(x, reps)
    })
}

export function batchDot(x, y, axis) {
    return tf.tidy(() => {
        return tf.sum(tf.mul(x, y), axis)
    })
}

export function mergeShape(tensor = tf.tensor(), axises, at = null) {
    return tf.tidy(() => {
        let shape = tensor.shape
        let merge = -1
        let transposeShape = new Array(shape.length).fill(0).map((val, idx) => idx)
        let newShape = null
        if (axises && axises.length != 0) {
            newShape = []
            at = at || axises[0]
            if (axises.find((axis) => axis == at) != undefined) {
                transposeShape.splice(at, 1, axises.slice())
            } else {
                console.error("axis ${at} is not at axises")
            }

            axises.sort(function (a, b) { //由大到小排序
                if (a > b) return -1;
                if (a < b) return 1;
                return 0;
            });
            axises.forEach((axis) => {
                if (!Array.isArray(transposeShape[axis])) {
                    transposeShape.splice(axis, 1)
                }
            })
            transposeShape = transposeShape.flat()
            shape.forEach((val, idx) => {
                if (axises.find(axis => axis == idx) != undefined) {
                    if (idx == at) {
                        merge = 1
                        axises.forEach(axis => {
                            merge *= shape[axis]
                        })
                        newShape.push(merge)
                    }
                } else {
                    newShape.push(val)
                }
            });
        }
        // console.log(transposeShape)
        // tensor.print()
        // tensor.transpose(transposeShape).print()
        return tensor.transpose(transposeShape).reshape(newShape || shape)
    })
}

export function matrixBandPart(input = tf.tensor(), numLower = 0, numUpper = 0) {
    return tf.tidy(() => {
        let [M, N] = [input.shape[input.shape.length - 2], input.shape[input.shape.length - 1]]
        let output = input.reshape([-1, M, N])
        let inBand = tf.tensor(new Array(M * N).fill(1).map((_, idx) => {
            let n = idx % N
            let m = (idx - n) / N
            return (numLower < 0 || (m - n) <= numLower) && (numUpper < 0 || (n - m) <= numUpper)
        }), [1, M, N])

        return tf.mul(output, inBand).reshape(input.shape)
    })
}

export let stopGradient = tf.customGrad((x, save) => {
    // Save x to make sure it's available later for the gradient.
    save([x])
    // Override gradient of our custom x ^ 2 op to be dy * abs(x);
    return {
        value: x.clone(),
        // Note `saved.x` which points to the `x` we saved earlier.
        gradFunc: (dy, saved) => {
            console.log("dy")
            dy.print()
            return [tf.mul(dy, 0)]
        }
    }
})

export function l2Normalize(x, axis = null, epsilon = 1e-12) {
    return tf.tidy(() => {
        let norm = tool.tensorPtr(tf.square(x))
            .sequence((tptr) => {
                tptr.assign(tf.sum(tptr.read(), axis, true))
                    .assign(tf.sqrt(tptr.read()))
            })

        let lower = tool.tensorPtr(tf.fill(norm.read().shape, epsilon))
        let isGreater = tool.tensorPtr(tf.greater(norm.read(), lower.read()))
        let output = tool.tensorPtr(tf.where(isGreater.read(), norm.read(), lower.read()))
        return output.assign(tf.div(x, output.read())).read()
    })
}

export function clipByGlobalNorm(tList, clipNorm) {
    return tf.tidy(() => {
        let newTList = tList.map(t => {
            return tf.tidy(() => {
                return tf.where(
                    t.isNaN(),
                    tf.fill(t.shape, Infinity),
                    t
                )
            })
        })
        let globalNorm = tool.tensorPtr()
            .sequence(gNptr => {
                gNptr.assign(
                    tf.addN(
                        newTList.map((t) => {
                            return tool.tensorPtr(tf.square(t))
                                .sequence(tptr => {
                                    tptr.assign(tf.sum(tptr.read()))
                                })
                                .read()
                        })
                    )
                )
                gNptr.assign(tf.sqrt(gNptr.read()))
            })
        return [
            newTList.map((t) => {
                let clip = tool.tensorPtr(tf.fill(globalNorm.read().shape, clipNorm))
                let isGreater = tool.tensorPtr(tf.greater(globalNorm.read(), clip.read()))
                let output = tool.tensorPtr(tf.mul(t, clipNorm))
                output.assign(tf.div(output.read(), tf.where(isGreater.read(), globalNorm.read(), clip.read())))
                return output.read()
            }),
            globalNorm.read()
        ]
    })
}

export function transpose(x, perm = null) {
    return tf.tidy(() => {
        let rankIndex = x.shape.map((_, index) => index)
        perm = perm ? perm : rankIndex.slice().sort((a, b) => { //由大到小排序
            if (a > b) return -1;
            if (a < b) return 1;
            return 0;
        })
        if (perm.length != x.shape.length) {
            console.error(`Error in transpose: rank of input ${x.shape.length} must match length of perm.`)
        } else if (perm.find((dim) => dim < 0 || dim > x.shape.length - 1)) {
            console.error(`All entries in 'perm' must be between 0 and ${x.shape.length - 1} but got ${perm}.`)
        }
        let _singleTranspose = (input, axis, perm, rankIndex) => {
            return tf.tidy(() => {
                let _rankIndex = rankIndex.slice()
                let moveTo = perm.indexOf(axis)
                let idx = _rankIndex.indexOf(axis)
                _rankIndex.splice(idx, 1)
                _rankIndex.splice(moveTo, 0, axis)

                let output = stack(
                    unstack(input, idx),
                    moveTo
                )
                return [output, _rankIndex]
            })
        }
        let output = tool.tensorPtr(x.clone())
        for (let i = 0; i < x.shape.length; i++) {
            [output.ptr, rankIndex] = _singleTranspose(output.read(), i, perm, rankIndex)
        }
        return output.read()
    })
}

export function stack(tensors = [tf.tensor()], axis) {
    return tf.tidy(() => {
        axis = axis != null ? axis : 0
        let shape = tensors[0].shape.slice()
        let strides = [shape[0] * (tensors[0].strides[0] || 1)].concat(tensors[0].strides).concat([1])
        if (tensors.find(tensor => !(tensor instanceof tf.Tensor) || (tensor.shape.toString() != shape.toString())) != undefined) {
            console.error(`All tensors passed to stack must match`)
            return
        }

        return shape.length == 0 ? tf.stack(tensors) :
            tf.tidy(() => {
                let output = tool.tensorPtr()
                shape.splice(axis, 0, tensors.length)
                return tool.tensorPtr(
                    tf.stack(
                        tensors.map(tensor => {
                            return tensor.reshape([-1, strides[axis]])
                        }),
                        1
                    )
                ).sequence(tptr => {
                    tptr.assign(tptr.read().reshape(shape))
                }).read()
            })
    })
}

export function unstack(x = tf.tensor(), axis) {
    return tf.tidy(() => {
        axis = axis != null ? axis : 0
        let shape = x.shape.slice()
        let strides = [shape[0] * (x.strides[0] || 1)].concat(x.strides).concat([1])
        if (!(x instanceof tf.Tensor)) {
            console.error(`x must be tensor`)
            return
        }

        return tf.unstack(
            x.reshape([-1, shape.splice(axis, 1)[0], strides[axis + 1]]),
            1
        ).map(t => {
            let tptr = tool.tensorPtr(t)
            return tptr.assign(tptr.read().reshape(shape)).read()
        })
    })
}

export function einsum(subscripts = "", ...operands) {
    return tf.tidy(() => {
        let subscript = {
            inputs: null,
            output: null
        };
        let _;

        [, subscript.inputs, _, subscript.output] = subscripts.match("^([a-zA-Z,.]+)(->)?([a-zA-Z.]*)?$") || [null, null, null, null]

        if (_ == null) {
            console.error(`Need "->"`)
            return
        }

        if (!subscript.inputs) {
            console.error(`Indices have incorrect format: ${subscripts}`)
            return
        }

        if (operands.find(input => !(input instanceof tf.Tensor)) != undefined || operands.length == 0) {
            console.error(`operands type is not tensor`)
            return
        }

        subscript.inputs = subscript.inputs.split(",")
        if (subscript.inputs.find(val => val == "") != undefined) {
            console.error(`Indices have incorrect format: ${subscripts}`)
            return
        }

        if (subscript.inputs.length != operands.length) {
            console.error(`Incorrect number of operands`)
            return
        }

        if (subscript.output == null) {
            subscript.output = ""
        }

        return _einsum(subscript, operands)
    })
}

function _einsum(subscript = { inputs: [""], output: null }, operands = [tf.tensor()]) {
    return tf.tidy(() => {
        let inputAnalysis = operands.map((t, idx) => {
            let axisTag = subscript.inputs[idx].split("")
            return t.shape.map((dim, axis) => {
                return { tag: axisTag[axis], axis: axis, dim: dim }
            }).sort((a, b) => { //由小到大排序
                if (a.tag > b.tag) return 1;
                if (a.tag < b.tag) return -1;
                return 0;
            })
        })

        let flatten = inputAnalysis.flat()
            .sort((a, b) => { //由小到大排序
                if (a.tag > b.tag) return 1;
                if (a.tag < b.tag) return -1;
                return 0;
            })
            .reduce((last, curr, index) => {
                if (last.length == 0) {
                    last.push(curr)
                    return last
                } else if (last[last.length - 1].tag != curr.tag) {
                    last.push(curr)
                    return last
                }
                return last
            }, [])

        let transposeTable = subscript.output.split("")
            .sort((a, b) => { //由小到大排序
                if (a > b) return 1;
                if (a < b) return -1;
                return 0;
            }).reduce((last, tag, axis) => {
                last[tag] = axis
                return last
            }, {})

        return tool.tensorPtrList()
            .sequence(tptrL => {//transpose
                tptrL.assign({ "output": tf.scalar(1) })
                for (let i = 0; i < operands.length; i++) {
                    tptrL.assign({ [`${i}`]: operands[i].transpose(inputAnalysis[i].map(e => e.axis)) })//transpose
                    let [newShape, reps] = [[], []]
                    for (let j = 0; j < flatten.length; j++) {
                        if (inputAnalysis[i].find((val) => val.tag == flatten[j].tag) != undefined) {
                            newShape.push(flatten[j].dim)
                            reps.push(1)
                        } else {
                            newShape.push(1)
                            reps.push(flatten[j].dim)
                        }

                    }
                    tptrL.assign({ [`${i}`]: tptrL.read(`${i}`).reshape(newShape) })//expand dims
                    tptrL.assign({ [`${i}`]: tile(tptrL.read(`${i}`), reps) })//repeat element
                    tptrL.assign({ "output": tf.mul(tptrL.read("output"), tptrL.read(`${i}`)) })//mul
                    tptrL.assign({ [`${i}`]: null })//dispose
                }
                tptrL.assign({
                    "output": tptrL.read("output").sum(
                        flatten
                            .reduce((last, curr, idx) => {
                                if (subscript.output.search(curr.tag) == -1) {
                                    last.push(idx)
                                }
                                return last
                            }, [])
                    )
                })//sum

                tptrL.assign({
                    "output": tptrL.read("output").transpose(
                        subscript.output.split("")
                            .map((tag) => {
                                return transposeTable[tag]
                            })
                    )
                })//transpose
            }).read("output")
    })
}

export function tile(x, reps) {
    return tf.tidy(() => {
        if (!(x instanceof tf.Tensor)) {
            console.error(`x must be tensor`)
            return
        }
        if (!(reps instanceof Array)) {
            console.error(`reps must be Number Array`)
            return
        } else if (reps.find(rep => typeof (rep) != "number") != undefined) {
            console.error(`reps must be Number Array`)
            return
        }

        let output = tool.tensorPtr(x.clone())
        for (let i = 0; i < reps.length; i++) {
            let strides = [output.read().shape[0] * output.read().strides[0], ...output.read().strides, 1]
            let newShape = output.read().shape.slice()
            newShape[i] *= reps[i]
            output.assign(output.read().reshape([-1, output.read().shape[i], strides[i + 1]]))
                .assign(output.read().tile([1, reps[i], 1]))
                .assign(output.read().reshape(newShape))
        }
        return output.read()
    })
}

export function softmax(logits, dim = -1) {
    return tf.tidy(() => {
        if (!(logits instanceof tf.Tensor)) {
            console.error(`logits must be a tensor`)
            return
        }
        dim = dim == -1 ? logits.shape.length - 1 : dim
        return tf.div(tf.exp(logits), tf.sum(tf.exp(logits), dim, true))
    })
}

export function softmaxCrossEntropyWithLogits(logits, labels, dim = -1) {
    return tf.tidy(() => {
        if (!(logits instanceof tf.Tensor)) {
            console.error(`logits must be a tensor`)
            return
        } else if (!(labels instanceof tf.Tensor)) {
            console.error(`labels must be a tensor`)
            return
        }
        dim = dim == -1 ? logits.shape.length - 1 : dim
        let y = softmax(logits, dim)
        let h = tf.mul(-1, tf.mul(labels, tf.log(y)).sum(dim))
        return h
    })
}