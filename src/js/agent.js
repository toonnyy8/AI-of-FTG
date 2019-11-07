import * as tf from "@tensorflow/tfjs"
import * as transformerXL from "./MirageNet/transformerXL_sp"
import { registerTfex } from "../lib/tfjs-extensions/src"
const tfex = registerTfex(tf)
// //載入權重
// import * as fs from "fs"
// let weight = fs.readFileSync(__dirname + "/../param/w.bin")
// tfex.scope.variableScope("transformerXL").load(tfex.sl.load(weight))
// console.log(tfex.scope.variableScope("transformerXL"))

tf.setBackend("webgl")

tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    let outputs = transformerXL.predict(
                        {
                            inp: tf.unstack(tf.stack(e.data.args.inps.map(val => tf.tensor(val)), 2), 0),
                            tgt: tf.unstack(tf.stack(e.data.args.tgts.map(val => tf.tensor(val)), 2), 0),
                            nToken: e.data.args.nToken,
                            FLAGS: e.data.args.FLAGS,
                            initializer: e.data.args.FLAGS.init == "normal" ?
                                tf.initializers.randomNormal({
                                    stddev: e.data.args.FLAGS.initStd
                                }) :
                                tf.initializers.randomUniform({
                                    minval: -1 * e.data.args.FLAGS.initRange,
                                    maxval: e.data.args.FLAGS.initRange
                                }),
                            projInitializer: tf.initializers.randomNormal({
                                stddev: e.data.args.FLAGS.initStd
                            }),
                            isTraining: true
                        }
                    )
                    outputs.forEach((output) => {
                        // console.log(output.shape)
                        // output.print()
                    })
                    tf.stack(outputs).array().then((d) => {
                        channel.postMessage({ instruction: "ctrl", output: d[d.length - 1] })
                        tf.dispose(outputs)
                    })
                    break
                }
                case 'train': {
                    transformerXL.train(
                        tf.unstack(tf.stack(e.data.args.inps.map(val => tf.tensor(val)), 2), 0),
                        tf.unstack(tf.stack(e.data.args.tgts.map(val => tf.tensor(val)), 2), 0),
                        e.data.args.nToken,
                        e.data.args.FLAGS,
                        e.data.args.FLAGS.init == "normal" ?
                            tf.initializers.randomNormal({
                                stddev: e.data.args.FLAGS.initStd
                            }) :
                            tf.initializers.randomUniform({
                                minval: -1 * e.data.args.FLAGS.initRange,
                                maxval: e.data.args.FLAGS.initRange
                            }),
                        tf.initializers.randomNormal({
                            stddev: e.data.args.FLAGS.initStd
                        }),
                        true,
                        e.data.args.bsz
                    )
                    channel.postMessage({ instruction: "train" })
                    break
                }
            }
        })
    }
})

document.getElementById("save").onclick = () => {
    tf.tidy(() => {
        let blob = new Blob([tfex.sl.save(tfex.scope.variableScope("transformerXL").save())]);
        let a = document.createElement("a");
        let url = window.URL.createObjectURL(blob);
        let filename = "w.bin";
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    })
}