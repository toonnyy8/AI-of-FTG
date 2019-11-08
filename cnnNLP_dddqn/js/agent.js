import * as tf from "@tensorflow/tfjs"
import { dddqn } from "../../src/js/MirageNet/dddqn"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

tf.setBackend("webgl")

let actionsNum = [3, 3, 4]

let dddqnModel = dddqn({
    sequenceLen: 16,
    stateVectorLen: 55,
    layerNum: 16,
    actionsNum: actionsNum,
    memorySize: 3200,
    minLearningRate: 5e-4,
    updateTargetStep: 0.1
})

let preArchive = {
    "player1": {
        state: null,
        actions: null,
        expired: true
    },
    "player2": {
        state: null,
        actions: null,
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
                        outputActions = outputActions.map(outputAction => {
                            outputAction = tf.softmax(outputAction, 1)
                            outputAction = tf.div(
                                tf.add(
                                    outputAction,
                                    1 / outputAction.shape[1]
                                ),
                                2
                            )
                            // outputAction.sum(1).print()
                            // outputAction.print()
                            return outputAction
                        })

                        let actions = []
                        let chooseByArgMax = outputActions.map(outputAction => {
                            return outputAction
                                .argMax(1)
                                .reshape([-1])
                                .arraySync()
                        })

                        let chooseByMultinomial = outputActions.map(outputAction => {
                            return tf.multinomial(outputAction, 1, null, true)
                                .reshape([-1])
                                .arraySync()
                        })
                        e.data.args.chooseActionRandomValue.forEach((chooseActionRandomValue, idx) => {
                            if (Math.random() < chooseActionRandomValue) {
                                actions[idx] = actionsNum.map((value, actionType) => {
                                    return chooseByMultinomial[actionType][idx]
                                })
                            } else {
                                actions[idx] = actionsNum.map((value, actionType) => {
                                    return chooseByArgMax[actionType][idx]
                                })
                            }
                        })

                        Object.keys(preArchive).forEach((playerName) => {
                            if (Object.keys(e.data.args.archive).find(name => name === playerName) !== undefined) {
                                if (preArchive[playerName].expired == false) {
                                    dddqnModel.store(
                                        preArchive[playerName].state,
                                        preArchive[playerName].actions,
                                        e.data.args.archive[playerName].rewards,
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
                            preArchive[playerName].actions = actions[idx]
                        })
                        channel.postMessage({
                            instruction: "ctrl",
                            args: {
                                archive: Object.keys(e.data.args.archive).reduce((acc, name, idx) => {
                                    acc[name] = {
                                        actions: actions[idx]
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