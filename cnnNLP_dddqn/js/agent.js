import * as tf from "@tensorflow/tfjs"
import { dddqn } from "../../src/js/MirageNet/dddqn"
import * as tfex from "../../src/lib/tfjs-extensions/src"

// //載入權重
// import * as fs from "fs"
// let weight = fs.readFileSync(__dirname + "/../param/w.bin")
// tfex.scope.variableScope("transformerXL").load(tfex.sl.load(weight))
// console.log(tfex.scope.variableScope("transformerXL"))

tf.setBackend("webgl")
let dddqnModel = dddqn({
    sequenceLen: 60,
    inputNum: 18,
    embInner: [32, 32, 32],
    filters: [8, 8, 8, 8, 8, 8, 8, 8, 64],
    outputInner: [32, 32],
    actionNum: 8
})
let preStates
let preActions
let preOutputs = tf.fill([1, 8], 1e-5)

let selectAction = (outputs) => {
    return tf.multinomial(outputs, 1, null, true)
}

tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    let outputs = dddqnModel
                        .model
                        .predict(tf.tensor(e.data.args.states))
                    // outputs = outputs.pow(0.5)
                    // outputs = tf.div(outputs, outputs.sum(1, true))
                    outputs.sub(preOutputs).div(preOutputs).array().then(a => console.log(a))

                    outputs.sub(preOutputs).div(preOutputs).argMax(1)
                        // selectAction(outputs)
                        .reshape([-1])
                        .array()
                        .then((actions) => {
                            if (preStates != null) {
                                preStates.forEach((s, idx) => {
                                    dddqnModel.store(e.data.args.states[idx], preActions[idx], e.data.args.rewards[idx], preStates[idx])
                                })
                            }
                            preStates = e.data.args.states
                            preActions = actions
                            channel.postMessage({ instruction: "ctrl", output: actions })
                        })
                    tf.dispose(preOutputs)
                    preOutputs = tf.keep(outputs)
                    console.log("ctrl")
                    break
                }
                case 'train': {
                    dddqnModel.train(10)
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

document.getElementById("selectAction").onclick = () => {
    if (document.getElementById("selectAction").innerText == "change to argMax") {
        selectAction = (outputs) => {
            return tf.argMax(outputs, 1)
        }
        document.getElementById("selectAction").innerText = "change to multinomial"
        document.getElementById("selectActionText").innerText = "select action : argMax"
    } else if (document.getElementById("selectAction").innerText == "change to multinomial") {
        selectAction = (outputs) => {
            return tf.multinomial(outputs, 1, null, true)
        }
        document.getElementById("selectAction").innerText = "change to argMax"
        document.getElementById("selectActionText").innerText = "select action : multinomial"
    }
}