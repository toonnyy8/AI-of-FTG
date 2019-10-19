import * as tf from "@tensorflow/tfjs"
import { dddqn } from "../../src/js/MirageNet/dddqn"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

//載入權重
import * as fs from "fs"
let saveWeights = fs.readFileSync(__dirname + "/../w.bin")
saveWeights = tfex.sl.load(saveWeights)

tf.setBackend("webgl")
let dddqnModel = dddqn({
    sequenceLen: 1024,
    inputNum: 18,
    embInner: [32, 32, 32],
    // filters: [8, 8, 8, 8, 8, 8, 8, 8, 64],
    filters: 64,
    outputInner: [32, 32],
    actionNum: 8
})
dddqnModel.model.getWeights().forEach((w) => {
    w.assign(saveWeights[w.name])
})

let preArchive = {
    "player1": {
        state: null,
        ASV: tf.fill([8], 1e-5, "float32"),
        preASV: tf.fill([8], 1e-5, "float32"),
        action: null,
        expired: true
    },
    "player2": {
        state: null,
        ASV: tf.fill([8], 1e-5, "float32"),
        preASV: tf.fill([8], 1e-5, "float32"),
        action: null,
        expired: true
    }
}

tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'ctrl': {
                    if (Object.keys(e.data.args.archive).length != 0) {
                        let ASVsAndActions = dddqnModel
                            .model
                            .predict([
                                tf.tensor(
                                    Object.values(e.data.args.archive)
                                        .map(archive => {
                                            return archive.state
                                        })
                                ),
                                tf.stack(
                                    Object.keys(e.data.args.archive)
                                        .map(playerName => {
                                            return preArchive[playerName].ASV
                                        })
                                )
                            ])
                        ASVsAndActions[1].argMax(1).print()

                        let actions = ASVsAndActions[1].argMax(1)
                            // selectAction(outputs)
                            .reshape([-1])
                            .arraySync()

                        Object.keys(preArchive).forEach((playerName) => {
                            if (Object.keys(e.data.args.archive).find(name => name === playerName) !== undefined) {
                                if (preArchive[playerName].expired == false) {
                                    dddqnModel.store(
                                        preArchive[playerName].state,
                                        preArchive[playerName].preASV.arraySync(),
                                        preArchive[playerName].action,
                                        e.data.args.archive[playerName].reward,
                                        e.data.args.archive[playerName].state,
                                        preArchive[playerName].ASV.arraySync()
                                    )
                                }
                                preArchive[playerName].expired = false
                            } else {
                                preArchive[playerName].expired = true
                            }
                        })

                        Object.keys(e.data.args.archive).forEach((playerName, idx) => {
                            preArchive[playerName].state = e.data.args.archive[playerName].state
                            tf.dispose(preArchive[playerName].preASV)
                            preArchive[playerName].preASV = tf.keep(preArchive[playerName].ASV)
                            preArchive[playerName].ASV = tf.keep(tf.unstack(ASVsAndActions[0])[idx])
                            preArchive[playerName].action = actions[idx]
                        })
                        channel.postMessage({
                            instruction: "ctrl",
                            args: {
                                archive: Object.keys(e.data.args.archive).reduce((acc, name, idx) => {
                                    acc[name] = {
                                        action: actions[idx]
                                    }
                                    return acc
                                }, {})
                            }
                        })
                    }
                    console.log("ctrl")
                    break
                }
                case 'train': {
                    dddqnModel.train(64)
                    channel.postMessage({ instruction: "train" })
                    break
                }
            }
        })
    }
})

document.getElementById("save").onclick = () => {
    tf.tidy(() => {
        let Ws = dddqnModel.model.getWeights()
        let tList = Ws.reduce((acc, w) => {
            acc[w.name] = w
            return acc
        }, {})
        console.log(tList)
        let blob = new Blob([tfex.sl.save(tList)]);
        let a = document.createElement("a");
        let url = window.URL.createObjectURL(blob);
        let filename = "w.bin";
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    })
}