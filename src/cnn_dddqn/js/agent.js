import * as tf from "@tensorflow/tfjs"
import { dddqn } from "../model"
import { registerTfex } from "../../lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

tf.setBackend("webgl")

let actionsNum = [2, 2, 2, 2, 2, 2, 2]

let dddqnModel = dddqn({
    sequenceLen: 4,
    stateVectorLen: 55,
    layerNum: 16,
    actionsNum: actionsNum,
    memorySize: 6400,
    minLearningRate: 1e-4,
    initLearningRate: 1e-3,
    updateTargetStep: 0.001,
    discount: 0.9
})

let preArchives = {
    "player1": {
        state: null,
        expired: true
    },
    "player2": {
        state: null,
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
                    if (Object.keys(e.data.args.archives).length != 0) {
                        let outputActions = dddqnModel
                            .model
                            .predict(
                                tf.transpose(
                                    tf.tensor(
                                        Object.values(e.data.args.archives)
                                            .map(archive => {
                                                return archive.state
                                            })
                                    ), [0, 2, 3, 1])
                            )
                        if (actionsNum.length == 1) {
                            outputActions = [outputActions]
                        }
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

                        let actions = {}
                        let chooseByArgMax = tf.concat(outputActions)
                            .argMax(1)
                            .reshape([outputActions.length, -1])
                            .transpose([1, 0])
                            .arraySync()

                        let chooseByMultinomial = tf.multinomial(tf.concat(outputActions), 1, null, true)
                            .reshape([outputActions.length, -1])
                            .transpose([1, 0])
                            .arraySync()

                        Object.keys(e.data.args.archives)
                            .forEach((playerName, idx) => {
                                if (Math.random() < e.data.args.archives[playerName].chooseActionRandomValue) {
                                    actions[playerName] = chooseByMultinomial[idx]
                                } else {
                                    actions[playerName] = chooseByArgMax[idx]
                                }
                            })

                        Object.keys(preArchives).forEach((playerName) => {
                            if (Object.keys(e.data.args.archives).find(name => name === playerName) !== undefined) {
                                if (preArchives[playerName].expired == false) {
                                    dddqnModel.store(
                                        preArchives[playerName].state,
                                        e.data.args.archives[playerName].actions,
                                        e.data.args.archives[playerName].rewards,
                                        e.data.args.archives[playerName].state,
                                    )
                                }
                                preArchives[playerName].expired = false
                            } else {
                                preArchives[playerName].expired = true
                            }
                        })

                        Object.keys(e.data.args.archives).forEach((playerName, idx) => {
                            preArchives[playerName].state = e.data.args.archives[playerName].state
                        })
                        channel.postMessage({
                            instruction: "ctrl",
                            args: {
                                archives: Object.keys(e.data.args.archives).reduce((acc, playerName) => {
                                    acc[playerName] = {
                                        actions: actions[playerName],
                                        aiCtrl: e.data.args.archives[playerName].aiCtrl
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
                case "updatePrioritys": {
                    dddqnModel.updatePrioritys()
                    channel.postMessage({ instruction: "updatePrioritys" })
                }
            }
        })
    }
})