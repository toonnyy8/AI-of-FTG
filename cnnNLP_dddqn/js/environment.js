import { Game } from "../../src/lib/slime-FTG/src/js"
import * as tf from "@tensorflow/tfjs"
import * as tfex from "../../src/lib/tfjs-extensions/src"
import * as FLAGS from "../../src/param/flags.json"

tf.setBackend("webgl")
// tf.enableProdMode()

export class Environment {
    constructor(
        players = [{
            name: "player1",
            actor: new Game().player1,
            keySet: {
                jump: "w",
                squat: "s",
                left: "a",
                right: "d",
                attack: {
                    small: "j",
                    medium: "k",
                    large: "l"
                }
            }
        }],
        memorySize = 256,
        ctrlLength = 5
    ) {
        this.memorySize = memorySize
        this.ctrlLength = ctrlLength
        this.players = players.reduce((last, player) => {
            last[player["name"]] = {
                actor: player["actor"],
                keySet: Object.keys(player["keySet"])
                    .reduce((last, actionName) => {
                        if (actionName == "attack") {
                            last[actionName] = Object.keys(player["keySet"]["attack"])
                                .reduce((last, attackName) => {
                                    last[attackName] = {
                                        keyup: new KeyboardEvent("keyup", {
                                            key: player["keySet"]["attack"][attackName]
                                        }),
                                        keydown: new KeyboardEvent("keydown", {
                                            key: player["keySet"]["attack"][attackName]
                                        })
                                    }
                                    return last
                                }, {})
                        } else {
                            last[actionName] = {
                                keyup: new KeyboardEvent("keyup", {
                                    key: player["keySet"][actionName]
                                }),
                                keydown: new KeyboardEvent("keydown", {
                                    key: player["keySet"][actionName]
                                })
                            }
                        }
                        return last
                    }, {}),
                memory: new Array(this.memorySize).fill(Environment.getMask()),
                rewardMemory: new Array(this.memorySize).fill(0),
                action: {
                    jump: false,
                    squat: false,
                    left: false,
                    right: false,
                    small: false,
                    medium: false,
                    large: false
                }
            }
            document.addEventListener('keydown', (event) => {
                Object.keys(player["keySet"]).forEach((actionName) => {
                    if (actionName == "attack") {
                        Object.keys(player["keySet"]["attack"]).forEach((actionName) => {
                            if (player["keySet"]["attack"][actionName] == event.key) {
                                last[player["name"]]["action"][actionName] = true
                            }
                        })
                    } else {
                        if (player["keySet"][actionName] == event.key) {
                            last[player["name"]]["action"][actionName] = true
                        }
                    }
                })
            })
            document.addEventListener('keyup', (event) => {
                Object.keys(player["keySet"]).forEach((actionName) => {
                    if (actionName == "attack") {
                        Object.keys(player["keySet"]["attack"]).forEach((actionName) => {
                            if (player["keySet"]["attack"][actionName] == event.key) {
                                last[player["name"]]["action"][actionName] = false
                            }
                        })
                    } else {
                        if (player["keySet"][actionName] == event.key) {
                            last[player["name"]]["action"][actionName] = false
                        }
                    }
                })
            })
            return last
        }, {})

        this.channel = new BroadcastChannel('agent');
        this.isReturnCtrl = true
        this.isReturnTrain = true
        this.channel.onmessage = (e) => {
            tf.tidy(() => {
                switch (e.data.instruction) {
                    case "ctrl":
                        {
                            this.isReturnCtrl = true
                            // console.log(e.data.output)
                            let output = e.data.output.pop()
                            let outputTensor = tf.tensor([
                                output[0].slice(1449, 1449 + 36),
                                output[1].slice(1449, 1449 + 36)
                            ])
                            outputTensor.sum(1, true).print()
                            outputTensor = tf.div(outputTensor, outputTensor.sum(1, true))
                            let action = tf.tidy(() => tf.multinomial(outputTensor, 1, null, true).add(1))
                            // let action = tf.tidy(() => tf.argMax(outputTensor, 1).add(1))
                            action.print()
                            action.array()
                                .then((aEnb) => {
                                    this.trigger(Object.keys(this.players)[0], actionDecoder(aEnb[0]))
                                    this.trigger(Object.keys(this.players)[1], actionDecoder(aEnb[1]))
                                    tf.dispose(outputTensor)
                                })
                            // tf.argMax(outputTensor, 1).add(1).print()
                            // tf.max(outputTensor, 1).print()
                            let { values, indices } = tf.topk(outputTensor, 36);
                            values.print();
                            indices.add(1).print();
                            // tf.argMax(outputTensor, 1).add(1).print()
                            // tf.argMax(outputTensor, 1).add(1).array()
                            //     .then((aEnb) => {
                            //         this.trigger(Object.keys(this.players)[0], actionDecoder(aEnb[0]))
                            //         this.trigger(Object.keys(this.players)[1], actionDecoder(aEnb[1]))
                            //         console.log(aEnb)
                            //     })
                            break
                        }
                    case "train":
                        {
                            this.isReturnTrain = true
                            break
                        }
                    default:
                        break;
                }
            })
        }
    }

    trigger(actorName, decodeAction) {
        switch (decodeAction[0]) {
            case 0:
                {
                    console.log(this.players[actorName].keySet["left"].keydown.key)

                    document.dispatchEvent(
                        this.players[actorName].keySet["left"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["left"].keydown
                    )
                    break;
                }
            case 1:
                {
                    console.log(this.players[actorName].keySet["right"].keydown.key)

                    document.dispatchEvent(
                        this.players[actorName].keySet["right"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["right"].keydown
                    )
                    break;
                }
            default:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["right"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["left"].keyup
                    )
                    break;
                }
        }
        switch (decodeAction[1]) {
            case 0:
                {
                    console.log(this.players[actorName].keySet["jump"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keydown
                    )
                    break;
                }
            case 1:
                {
                    console.log(this.players[actorName].keySet["squat"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keydown
                    )
                    break;
                }
            default:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keyup
                    )
                    break;
                }
        }
        switch (decodeAction[2]) {
            case 1:
                {
                    console.log(this.players[actorName].keySet.attack["small"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keydown
                    )
                    break;
                }
            case 2:
                {
                    console.log(this.players[actorName].keySet.attack["medium"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keydown
                    )
                    break;
                }
            case 3:
                {
                    console.log(this.players[actorName].keySet.attack["large"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keydown
                    )
                    break;
                }
            default:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keyup
                    )
                    break;
                }
        }
    }

    fetchUpReward() {
        Object.keys(this.players).forEach((playerName) => {
            this.players[playerName]["rewardMemory"].push(getReward(this.players[playerName]["actor"]))
            if (this.players[playerName]["rewardMemory"].length > this.memorySize) {
                this.players[playerName]["rewardMemory"].shift()
            }
        })
    }

    nextStep() {
        Object.keys(this.players).forEach((playerName) => {
            this.players[playerName]["memory"].unshift(
                Environment.getState(this.players[playerName]["actor"], this.players[playerName]["actor"].opponent)
            )
            if (this.players[playerName]["memory"].length > this.memorySize) {
                this.players[playerName]["memory"].pop()
            }
        })
    }

    control(playersName) {
        let inps = playersName.map((playerName) => {
            return [this.players[playerName]["memory"].slice(0, ctrlLength)]
        })
        console.log(inps)
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                state: inps,
                reward: inps,
            }
        })
        console.log("ctrl")
    }

    mergeMemory(mainPlayerName, mergeLength, end = this.players[mainPlayerName]["memory"].length - 1) {
        let mergeMem = []
        let mergeRewardMem = []
        end = end <= this.players[mainPlayerName]["memory"].length - 1 ? end : this.players[mainPlayerName]["memory"].length - 1
        mergeLength = end - mergeLength >= 0 ? mergeLength : end
        for (let i = end - mergeLength + 1; i <= end; i++) {
            mergeMem.push(this.players[mainPlayerName]["memory"][i].slice())
            mergeRewardMem.push(this.players[mainPlayerName]["rewardMemory"][i])
            Object.keys(this.players).forEach((playerName) => {
                if (playerName != mainPlayerName) {
                    mergeMem.push(this.players[playerName]["memory"][i])
                    mergeRewardMem.push(this.players[playerName]["rewardMemory"][i])
                }
            })
        }
        return [mergeMem, mergeRewardMem]
    }

    predictReward(playerName) {
        return this.players[playerName]["rewardMemory"].reduce((pReward, reward) => {
            return (pReward + reward) * 0.5
        }, 0)
    }

    train(simulationBsz = 1, bsz = 1) {
        let rewards = []
        let origins = []
        for (let i = 0; i < simulationBsz; i++) {
            origins = origins.concat(
                Object.keys(this.players).map((playerName) => {
                    let end = Math.round(Math.random() * (this.memorySize - this.ctrlLength - 1) + this.ctrlLength)
                    let [origin, reward] = this.mergeMemory(playerName, this.ctrlLength + 1, end)
                    origin = origin.slice(1, origin.length - 1)
                    reward = reward.slice(1, origin.length - 1)
                    rewards.push(reward.map(r => r + 6 + 1484))
                    return [origin.flat()]
                })
            )
        }
        // console.log(origins)
        let inps = origins.map((origin, originIdx) => {
            return [origin.map((words) => {
                let inp = []
                for (let i = 0; i < words.length; i++) {
                    inp.push(words[i])
                }
                inp.pop()
                inp.push(rewards[originIdx][rewards[originIdx].length - 1])
                return inp
            }).flat()]
        })
        // console.log(inps)
        let tgts = origins.map((origin, originIdx) => {
            return [origin.map((words) => {
                let tgt = []
                for (let i = 0; i < words.length; i++) {
                    tgt.push(words[i])
                }
                tgt.pop()
                tgt.push(words[words.length - 1])
                return tgt
            }).flat()]
        })
        // console.log(tgts)
        this.channel.postMessage({
            instruction: "train",
            args: {
                inps: inps,
                tgts: tgts,
                nToken: 1496,
                FLAGS: FLAGS,
                bsz: bsz
            }
        })
        console.log("train")
    }

    static getState(actorA, actorB) {
        //actorA state
        let getS = (actor) => {
            let HP = actor.HP
            let x = actor.mesh.position.x
            let y = actor.mesh.position.y
            let faceTo = actor._faceTo == "left" ? 0 : 1

            let chapter
            switch (actor._state["chapter"]) {
                case "normal": {
                    chapter = 0
                    break
                } case "attack": {
                    chapter = 1
                    break
                } case "defense": {
                    chapter = 2
                    break
                } case "hitRecover": {
                    chapter = 3
                    break
                }
            }

            let section
            switch (actor._state["section"]) {
                case "stand": {
                    section = 0
                    break
                } case "jump": {
                    section = 1
                    break
                } case "squat": {
                    section = 2
                    break
                } case "reStand": {
                    section = 3
                    break
                }
            }

            let subsection
            switch (actor._state["subsection"]) {
                case "main": {
                    subsection = 0
                    break
                } case "forward": {
                    subsection = 1
                    break
                } case "backward": {
                    subsection = 2
                    break
                } case "small": {
                    subsection = 10
                    break
                } case "medium": {
                    subsection = 20
                    break
                } case "large": {
                    subsection = 30
                    break
                } case "fall": {
                    subsection = 40
                    break
                }
            }

            let subsubsection
            switch (actor._state["subsubsection"]) {
                case "0": {
                    subsubsection = 0
                    break
                } case "1": {
                    subsubsection = 1
                    break
                } case "2": {
                    subsubsection = 2
                    break
                } case "3": {
                    subsubsection = 3
                    break
                }
            }

            return [HP, x, y, faceTo, chapter, section, subsection, subsubsection]
        }

        return getS(actorA).concat(getS(actorB))
    }

    static getMask() {
        let getM = () => {
            return [-1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30]
        }

        return getM().concat(getM())
    }

    static getReward(actor) {
        let reward = Math.round((actor.HP - actor.opponent.HP) / 1500)
        if (actor.isPD) {
            reward += 5
        }
        if (actor.isHit) {
            reward += actor.opponent.beHitNum
        }
        if (actor._state.chapter == "defense") {
            reward += 3
        }
        if (actor.beHitNum != 0) {
            reward -= actor.beHitNum
        }

        reward = Math.min(Math.max(reward, -5), 5)

        return reward
    }

    static actionDecoder(encodeAction) {
        let lr = Math.floor(encodeAction / 12)
        let js = Math.floor((encodeAction - (12 * lr)) / 4)
        let atk = encodeAction - lr * 12 - js * 4
        return [lr, js, atk]
    }

    static getAction(actor) {
        let leftOrRight = action["left"] ? 0 : action["right"] ? 1 : 2
        let jumpOrSquat = action["jump"] ? 0 : action["squat"] ? 1 : 2
        let attack = action["small"] ? 0 : action["medium"] ? 1 : action["large"] ? 2 : 3
        return (leftOrRight * 3 + jumpOrSquat) * 4 + attack
    }

}