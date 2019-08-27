import "@babel/polyfill"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"

// (async() => {
//     console.log(tf.memory())
//     let a = tf.tensor([
//         [
//             [1, 4, 2],
//             [1, 2, 1]
//         ],
//         [
//             [1, 4, 3],
//             [1, 3, 1]
//         ]
//     ])
//     let b = tf.tensor([
//         [
//             [1, 3],
//             [2, 1],
//             [3, 3]
//         ],
//         [
//             [1, 1],
//             [1, 2],
//             [2, 4]
//         ]
//     ])

//     console.log("---------")
//     console.log("tf.einsum('ijk,gkh->k',a,b) : [24, 78, 84]")
//     await testTime(() => {
//         tfex.einsum('ijk,gkh->k', a, b).print()
//     }, "tf.einsum('ijk,gkh->k',a,b)")
//     console.log("---------")
//     console.log("tf.einsum('ijk,imj->i',a,b) : [70 67]")
//     await testTime(() => {
//         tfex.einsum('ijk,imj->i', a, b).print()
//     }, "tf.einsum('ijk,imj->i',a,b)")
//     console.log("---------")
//     console.log("tf.einsum('ijk,iml->i',a,b) : [143 143]")
//     await testTime(() => {
//         tfex.einsum('ijk,iml->', a, b).print()
//     }, "tf.einsum('ijk,iml->',a,b)")
//     console.log("---------")
//     await testTime(() => {
//         tfex.einsum('ijk,imj->ij', a, b).print()
//     }, "tf.einsum('ijk,imj->ij',a,b)")
//     console.log("---------")
//     console.log(tf.memory())

// })()

// async function testTime(f = () => {}, msg = "msg") {
//     const time = await tf.time(f)
//     console.log(`${msg}--kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
// }

// let a = tfex.layers.lambda({ func: (x, y) => { return [tf.add(x, y), tf.add(x, y), tf.add(x, y)] } })
// console.log(tf.memory())
// console.log(
//     a.apply([tf.input({ shape: [3] }), tf.input({ shape: [3] })])
// )
// console.log(tf.memory())

console.log(tf.memory())

let a = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
let b = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
tfex.l2Normalize(a, 0).print()
tfex.l2Normalize(a, 1).print()
tfex.l2Normalize(a).print()
tfex.clipByGlobalNorm([a, a], 0.001)[1].print()
tfex.clipByGlobalNorm([a, a], 0.001)[0][0].print()
tfex.clipByGlobalNorm([a, a], 0.1)[0][0].print()
    // tf.tidy(() => {
    //     tfex.scope.getVariable("1", [3])

//     tf.tidy(() => {
//         let FF = tfex.scope.variableScope("FF")
//         for (let i = 0; i < 6; i++) {
//             FF.getVariable(`layer_${i}`, [1, 1, 1])
//         }
//         tf.tidy(() => {
//             let pE = FF.variableScope("pE")
//             for (let i = 0; i < 2; i++) {
//                 pE.getVariable(`qkv_${i}`, [1, 1, 1])
//             }
//         })
//     })
// })
// // tfex.scope.variableScope("FF").dispose()
// let saveData = tfex.scope.save()
// tfex.scope.variableScope("test").load(saveData)
// console.log(tfex.scope.save())
// console.log(tfex.scope.allVariables())


console.log(tf.memory())