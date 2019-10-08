import * as tokenSet from "../param/tokens.json"
import { Game } from "../lib/slime-FTG/src/js"
import * as tf from "@tensorflow/tfjs"
import * as tfex from "../lib/tfjs-extensions/src"
import * as FLAGS from "../param/flags.json"

tf.setBackend("webgl")
// tf.enableProdMode()

export function getStatement(actor, actorName = "player1" || "player2", action) {
    //stateA
    let player = actorName == "player1" ? 0 : 500

    let faceTo = actor._faceTo == "left" ? 0 : 250

    let x = Math.abs(actor.mesh.position.x - actor.opponent.mesh.position.x)
    if (x < 1) {
        x = 0
    } else if (x < 2.5) {
        x = 50
    } else if (x < 4.5) {
        x = 100
    } else if (x < 7) {
        x = 150
    } else {
        x = 200
    }

    let y = actor.mesh.position.y
    if (y < 2) {
        y = 0
    } else if (y < 4) {
        y = 10
    } else if (y < 6) {
        y = 20
    } else if (y < 8) {
        y = 30
    } else {
        y = 40
    }

    let hp = actor.HP
    if (hp < 300) {
        hp = 1
    } else if (hp < 600) {
        hp = 2
    } else if (hp < 900) {
        hp = 3
    } else if (hp < 1200) {
        hp = 4
    } else if (hp < 1500) {
        hp = 5
    } else if (hp < 1800) {
        hp = 6
    } else if (hp < 2100) {
        hp = 7
    } else if (hp < 2400) {
        hp = 8
    } else if (hp < 2700) {
        hp = 9
    } else {
        hp = 10
    }

    //stateB
    let chapter = actor._state["chapter"]
    if (chapter == "normal") {
        chapter = 0
    } else if (chapter == "attack") {
        chapter = 112
    } else if (chapter == "defense") {
        chapter = 112 * 2
    } else if (chapter == "hitRecover") {
        chapter = 112 * 3
    }

    let section = actor._state["section"]
    if (section == "stand") {
        section = 0
    } else if (section == "jump") {
        section = 28
    } else if (section == "squat") {
        section = 28 * 2
    } else if (section == "reStand") {
        section = 28 * 3
    }

    let subsection = actor._state["subsection"]
    if (subsection == "main") {
        subsection = 0
    } else if (subsection == "forward") {
        subsection = 4
    } else if (subsection == "backward") {
        subsection = 8
    } else if (subsection == "small") {
        subsection = 12
    } else if (subsection == "medium") {
        subsection = 16
    } else if (subsection == "large") {
        subsection = 20
    } else if (subsection == "fall") {
        subsection = 24
    }

    let subsubsection = actor._state["subsubsection"]
    if (subsubsection == "0") {
        subsubsection = 1
    } else if (subsubsection == "1") {
        subsubsection = 2
    } else if (subsubsection == "2") {
        subsubsection = 3
    } else if (subsubsection == "3") {
        subsubsection = 4
    }

    //actions
    let leftOrRight = action["left"] ? 0 : action["right"] ? 12 : 24
    let jumpOrSquat = action["jump"] ? 0 : action["squat"] ? 4 : 8
    let attack = action["small"] ? 1 : action["medium"] ? 2 : action["large"] ? 3 : 4

    return [player + faceTo + x + y + hp, 1000 + chapter + section + subsection + subsubsection, 1448 + leftOrRight + jumpOrSquat + attack]
}

export function actionDecoder(encodeAction) {
    let lr = Math.ceil(encodeAction / 12) - 1
    let js = Math.ceil((encodeAction - (12 * lr)) / 4) - 1
    let atk = encodeAction - lr * 12 - js * 4
    return [lr, js, atk]
}

export function getReward(actor) {
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

export function maskAction(statement) {
    let cloneS = statement.slice()
    let opStart = cloneS.indexOf(tokenSet.tokens["<op>"])
    let opEnd = cloneS.indexOf(tokenSet.tokens["</op>"])
    for (let i = opStart + 1; i < opEnd; i++) {
        cloneS[i] = tokenSet.tokens["mask"]
    }
    return cloneS
}


export class Environment {
    constructor(players = [{
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
    }], memorySize = 256, ctrlLength = 5) {
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
                memory: [],
                rewardMemory: [],
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
                            indices.print();
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
            this.players[playerName]["memory"].push(
                getStatement(
                    this.players[playerName]["actor"],
                    playerName,
                    this.players[playerName]["action"]
                )
            )
            if (this.players[playerName]["memory"].length > this.memorySize) {
                this.players[playerName]["memory"].shift()
            }

        })
    }

    control(playersName) {
        let inps = playersName.map((playerName) => {
            let pReward = Math.min(Math.round(this.predictReward(playerName) + 1), 5)
            let newStatement = getStatement(this.players[playerName]["actor"], playerName, this.players[playerName]["action"])
            newStatement[2] = 1484 + 6 + pReward

            let [inp, _] = this.mergeMemory(playerName, this.ctrlLength)
            inp.push(newStatement)
            inp = inp.flat()
            if (inp.length > 3) {
                inp.shift()
                inp.shift()
                inp.shift()
            }
            return [inp]
        })
        console.log(inps)
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                inps: inps,
                tgts: inps,
                nToken: 1496,
                FLAGS: FLAGS
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
}