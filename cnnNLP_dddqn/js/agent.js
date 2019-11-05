import * as tf from "@tensorflow/tfjs"
import { dddqn } from "../../src/js/MirageNet/dddqn"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

tf.setBackend("webgl")

let actionNum = 9

let dddqnModel = dddqn({
    sequenceLen: 64,
    inputNum: 55,
    filters: 32,
    actionNum: actionNum,
    memorySize: 3200,
    minLearningRate: 5e-4,
    updateTargetStep: 32
})

let preArchive = {
    "player1": {
        state: null,
        action: null,
        expired: true
    },
    "player2": {
        state: null,
        action: null,
        expired: true
    }
}

tf.ready().then(() => {
    let channel = new BroadcastChannel('agent');
    channel.onmessage = (e) => {
        tf.tidy(() => {
            switch (e.data.instruction) {
                case 'init': {
                    channel.postMessage({ instruction: "init" })
                    break
                }
                case 'ctrl': {
                    if (Object.keys(e.data.args.archive).length != 0) {
                        let outputActions = dddqnModel
                            .model
                            .predict(
                                tf.tensor(
                                    Object.values(e.data.args.archive)
                                        .map(archive => {
                                            return archive.state
                                        })
                                )
                            )
                        // outputActions.sum(1).print()
                        // outputActions.print()

                        let actions = []
                        let chooseByArgMax = outputActions.argMax(1)
                            .reshape([-1])
                            .arraySync()
                        let chooseByMultinomial = tf.multinomial(outputActions, 1, null, true)
                            .reshape([-1])
                            .arraySync()
                        e.data.args.chooseAction.forEach((chooseAction, idx) => {
                            if (chooseAction == "argMax") {
                                actions[idx] = chooseByArgMax[idx]
                            } else if (chooseAction == "multinomial") {
                                actions[idx] = chooseByMultinomial[idx]
                            }
                        })


                        Object.keys(preArchive).forEach((playerName) => {
                            if (Object.keys(e.data.args.archive).find(name => name === playerName) !== undefined) {
                                if (preArchive[playerName].expired == false) {
                                    dddqnModel.store(
                                        preArchive[playerName].state,
                                        preArchive[playerName].action,
                                        e.data.args.archive[playerName].reward,
                                        e.data.args.archive[playerName].state,
                                    )
                                }
                                preArchive[playerName].expired = false
                            } else {
                                preArchive[playerName].expired = true
                            }
                        })

                        Object.keys(e.data.args.archive).forEach((playerName, idx) => {
                            preArchive[playerName].state = e.data.args.archive[playerName].state
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
                    } else {
                        channel.postMessage({
                            instruction: "ctrl",
                            args: {
                                archive: {}
                            }
                        })
                    }
                    // console.log("ctrl")
                    break
                }
                case 'train': {
                    dddqnModel.train(e.data.args.bsz, e.data.args.replayIdxes, e.data.args.usePrioritizedReplay)
                    channel.postMessage({ instruction: "train" })
                    break
                }
                case 'save': {
                    tf.tidy(() => {
                        let Ws = dddqnModel.model.getWeights()
                        let tList = Ws.reduce((acc, w) => {
                            acc[w.name] = w
                            return acc
                        }, {})
                        channel.postMessage({
                            instruction: "save",
                            args: {
                                weightsBuffer: tfex.sl.save(tList)
                            }
                        })
                    })

                    break
                }
                case 'load': {
                    let loadWeights = tfex.sl.load(e.data.args.weightsBuffer)
                    dddqnModel.model.getWeights().forEach((w) => {
                        w.assign(loadWeights[w.name])
                    })
                    dddqnModel.targetModel.setWeights(
                        dddqnModel.model.getWeights()
                    )
                    channel.postMessage({ instruction: "load" })
                    break
                }
            }
        })
    }
})