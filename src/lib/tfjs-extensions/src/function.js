import * as tf from "@tensorflow/tfjs"

export function repeatElements(x, rep, axis) {
    let reps = [...Array(x.shape.length)].map(_ => 1)
    reps[axis] = rep
    return tf.tile(x, reps)
}

export function batchDot(x, y, axis) {
    return tf.sum(tf.mul(x, y), axis)
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

            axises.sort(function(a, b) { //由大到小排序
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

export function einsum(subscripts = "", ...operands) {
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

    if (subscript.inputs.length == 1) {
        return einsumSingleInput(subscript, operands[0])
    } else {
        return einsumMultipleInput(subscript, operands)
    }
}

function einsumSingleInput(subscript = { inputs: [""], output: "" }, operand = tf.tensor()) {
    return tf.tidy(() => {
        let inputInfo = subscript.inputs[0]
            .split("")
            .map((tag, axis) => {
                return { tag: tag, axis: axis, dim: operand.shape[axis] }
            })
            .sort((a, b) => { //由小到大排序
                if (a.tag > b.tag) return 1;
                if (a.tag < b.tag) return -1;
                return 0;
            })

        let outputInfo = subscript.output == "" ? [] : subscript.output
            .split("")
            .map((tag, axis) => {
                if (subscript.inputs[0].search(tag) == -1) {
                    console.error(`Output Tag_${tag} does not exist`)
                }
                return { tag: tag, axis: axis }
            })
            .sort((a, b) => { //由小到大排序
                if (a.tag > b.tag) return 1;
                if (a.tag < b.tag) return -1;
                return 0;
            })

        let diadInfo = inputInfo.reduce((last, info) => {
            if (last[info.tag]) {
                if (last[info.tag][0] != info.dim) {
                    console.error(`Dim of Tag_${info.tag} are inconsistent`)
                } else {
                    last[info.tag].push(info.dim)
                }
            } else {
                last[info.tag] = [info.dim]
            }
            return last
        }, {})

        let diagShape = Object.values(diadInfo)

        let indices = diagIndices(diagShape)

        let tagSum = Object.keys(diadInfo)
            .reduce((last, tag, axis) => {
                if (subscript.output.search(tag) == -1) {
                    last.push(axis)
                }
                return last
            }, [])

        return operand.
        transpose(inputInfo.map((info) => info.axis))
            .reshape([-1])
            .gather(indices)
            .sum(tagSum)
            .transpose(outputInfo.map((info) => info.axis))
    })
}

function diagIndices(diagShape = [
    []
]) {
    let diagShape_ = JSON.parse(JSON.stringify(diagShape))
    let getDiagIndices = (dim, edgeNum) => {
        let stop = dim ** edgeNum
        let step = 1

        for (let i = 1; i < edgeNum; i++) {
            step = step * dim + 1
        }
        return [stop, step]
    }
    return tf.tidy(() => {
        let pre = 1
        let indices = tf.range(0, diagShape_.flat().reduce((last, dim) => last * dim, 1), 1, "int32")
        for (let i = 0; i < diagShape_.length; i++) {
            if (diagShape_[i].length > 1) {
                let [stop, step] = getDiagIndices(diagShape_[i][0], diagShape_[i].length)
                indices = indices
                    .reshape([pre, diagShape_[i].reduce((last, dim) => last * dim, 1), -1])
                    .gather(tf.range(0, stop, step, "int32"), 1)
            }
            pre *= diagShape_[i][0]
        }
        indices = indices.reshape(diagShape_.map((dim) => dim[0]))
        return indices
    })
}

function einsumMultipleInput(subscript = { inputs: [""], output: null }, operands = [tf.tensor()]) {
    return tf.tidy(() => {
        let inputInfo = {
            x: subscript.inputs
                .shift()
                .split("")
                .map((tag, axis) => {
                    return { tag: tag, axis: axis }
                })
                .sort((a, b) => { //由小到大排序
                    if (a.tag > b.tag) return 1;
                    if (a.tag < b.tag) return -1;
                    return 0;
                }),
            y: subscript.inputs
                .shift()
                .split("")
                .map((tag, axis) => {
                    return { tag: tag, axis: axis }
                })
                .sort((a, b) => { //由小到大排序
                    if (a.tag > b.tag) return 1;
                    if (a.tag < b.tag) return -1;
                    return 0;
                })
        }

        let [x, y] = [
            operands.shift().transpose(inputInfo.x.map((info) => info.axis)),
            operands.shift().transpose(inputInfo.y.map((info) => info.axis))
        ]

        operands.unshift(
            x
            .reshape([-1, 1])
            .dot(y.reshape([1, -1]))
            .reshape(x.shape.concat(y.shape))
        )
        subscript.inputs.unshift(
            inputInfo.x
            .reduce((last, info) => last + info.tag, "")
            .concat(
                inputInfo.y
                .reduce((last, info) => last + info.tag, "")
            )
        )

        if (subscript.inputs.length == 1) {
            return einsumSingleInput(subscript, operands[0])
        } else {
            return einsumMultipleInput(subscript, operands)
        }
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
        gradFunc: (dy, saved) => [tf.zeros(saved[0].shape)]
    }
})

export function l2Normalize(x, axis = null, epsilon = 1e-12) {
    return tf.tidy(() => {
        let norm = tf.sqrt(tf.sum(tf.square(x), axis, true))
        let lower = tf.fill(norm.shape, epsilon)
        let isGreater = tf.greater(norm, lower)
        return tf.div(x, tf.where(isGreater, norm, lower))
    })
}

export function clipByGlobalNorm(tList, clipNorm) {
    return tf.tidy(() => {
        let globalNorm = tf.sqrt(
            tf.addN(
                tList.map((t) => {
                    return tf.sum(tf.square(t))
                })
            )
        )
        return [
            tList.map((t) => {
                let lower = tf.fill(globalNorm.shape, clipNorm)
                let isGreater = tf.greater(globalNorm, lower)
                return tf.div(tf.mul(t, clipNorm), tf.where(isGreater, globalNorm, lower))
            }),
            globalNorm
        ]
    })
}