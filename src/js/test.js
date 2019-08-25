import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim
import * as FLAGS from "../other/flags.json"
import * as transformerXL from "./MirageNet/transformerXL"
console.log(transformerXL)
console.log(tf.memory())

const b = tf.tensor([-2, -3])
// f(a, b) = a * b
const f = () => {
    return a.add(tf.square(b))
};
let h = () => { return f() }
// df / da = b, df / db = a
const g = tf.grads(h);

const a = tf.tensor([2, 3])
const c = tf.tensor([-8, 3])
const [da, db] = g([a, b]);
console.log('da');
da.print();
console.log('db');
db.print();
// tf.grads(() => {
//     return transformerXL.transformer({
//         decInp: inp,
//         target: tgt,
//         mems: mems,
//         nToken: n_token,
//         nLayer: FLAGS.n_layer,
//         dModel: FLAGS.d_model,
//         dEmbed: FLAGS.d_embed,
//         nHead: FLAGS.n_head,
//         dHead: FLAGS.d_head,
//         dInner: FLAGS.d_inner,
//         dropout: FLAGS.dropout,
//         dropatt: FLAGS.dropatt,
//         initializer: initializer,
//         projInitializer: proj_initializer,
//         isTraining: is_training,
//         memLen: FLAGS.mem_len,
//         cutoffs: cutoffs,
//         divVal: FLAGS.div_val,
//         tieProjs: tie_projs,
//         inputPerms: None,
//         targetPerms: None,
//         headTarget: None,
//         sameLength: FLAGS.same_length,
//         clampLen: FLAGS.clamp_len,
//         useTpu: False,
//         untieR: FLAGS.untie_r,
//         projSameDim: FLAGS.proj_same_dim
//     },
//         tfex.scope.variableScope("transformerXL")
//     )
// })

console.log(tf.memory())