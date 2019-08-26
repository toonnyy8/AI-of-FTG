import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
// import * as tf from "@tensorflow/tfjs/dist/tf"// to spacevim
import * as FLAGS from "../other/flags.json"
import * as transformerXL from "./MirageNet/transformerXL"
console.log(transformerXL)
console.log(tf.memory())

const stopGradient = tf.customGrad((x, save) => {
    // Save x to make sure it's available later for the gradient.
    save([x]);
    // Override gradient of our custom x ^ 2 op to be dy * abs(x);
    return {
        value: x.clone(),
        // Note `saved.x` which points to the `x` we saved earlier.
        gradFunc: (dy, saved) => [tf.zeros(saved[0].shape)]
    };
});

const x = tf.tensor1d([-1, -2, 3]);

const f = (x) => x.add(stopGradient(x))
const dx = tf.grad(f);

console.log(`f(x):`);
f(x).print();
console.log(`f'(x):`);
dx(x).print();

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