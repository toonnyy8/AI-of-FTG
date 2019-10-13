import * as tf from "@tensorflow/tfjs"
import * as cnnNLP from "../../src/js/MirageNet/CNNNLP"
import * as tfex from "../../src/lib/tfjs-extensions/src"

// //載入權重
// import * as fs from "fs"
// let weight = fs.readFileSync(__dirname + "/../param/w.bin")
// tfex.scope.variableScope("transformerXL").load(tfex.sl.load(weight))
// console.log(tfex.scope.variableScope("transformerXL"))

tf.setBackend("webgl")
let model = cnnNLP.buildModel({
    sequenceLen: 60,
    inputNum: 10,
    embInner: [64, 64, 64],
    filters: 64,
    outputInner: [512, 512, 512],
    outputNum: 36
})
tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    let output = model.predict(e.data.x)
                    output.array().then((a) => {
                        channel.postMessage({ instruction: "ctrl", output: a })
                        tf.dispose(output)
                    })
                    break
                }
                case 'train': {

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