import * as tf from "@tensorflow/tfjs"
import * as transformerXL from "./MirageNet/transformerXL_sp"

tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    let outputs = transformerXL.modelFn(
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
                        })
                    )
                    outputs.forEach((output) => { output.print() })
                    tf.stack(outputs).array().then((d) => {
                        channel.postMessage({ instruction: "ctrl", output: d })
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
                        })
                    )
                    channel.postMessage({ instruction: "train" })
                    break
                }
            }
        })
    }
})