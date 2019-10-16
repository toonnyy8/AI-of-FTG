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
    filters: [8, 8, 8, 8, 64],
    outputInner: [32, 32],
    actionNum: 8
})
let preStates
let preActions
tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    let outputs = dddqnModel
                        .model
                        .predict(tf.tensor(e.data.args.states))
                    outputs.array().then(a => console.log(a))
                    tf.multinomial(outputs, 1, null, true)
                        .reshape([-1])
                        .array()
                        .then((actions) => {
                            if (preStates != null) {
                                preStates.forEach((s, idx) => {
                                    // dddqnModel.store(e.data.args.states[idx], e.data.args.actions[idx], e.data.args.rewards[idx], preStates[idx])
                                    dddqnModel.store(e.data.args.states[idx], preActions[idx], e.data.args.rewards[idx], preStates[idx])
                                })
                                // dddqnModel.train(1)
                            }
                            // console.log(preActions)
                            preStates = e.data.args.states
                            preActions = actions
                            channel.postMessage({ instruction: "ctrl", output: actions })
                        })

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