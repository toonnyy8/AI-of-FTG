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
    sequenceLen: 30,
    actionNum: 16,
    embInner: [64, 64, 64],
    filters: 64,
    outputInner: [512, 512, 512],
    actionNum: 36
})
let preStates
let preActions
tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    dddqnModel
                        .model
                        .predict(e.data.args.states)
                        .argMax(1)
                        .array()
                        .then((actions) => {
                            preStates.forEach((s, idx) => {
                                dddqnModel.store(e.data.args.states[idx], preActions[idx], e.data.args.rewards[idx], preStates[idx])
                            })

                            preStates = e.data.args.state
                            preActions = actions
                            channel.postMessage({ instruction: "ctrl", output: actions })
                        })
                    console.log("ctrl")
                    break
                }
                case 'train': {
                    dddqnModel.train(100)
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