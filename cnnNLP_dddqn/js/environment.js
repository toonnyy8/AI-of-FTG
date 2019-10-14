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
                memory: new Array(this.memorySize).fill(Environment.getState(player["actor"], player["actor"].opponent)),
                reward: 0,
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
                            let output = e.data.output
                            this.trigger(Object.keys(this.players)[0], Environment.actionDecoder(output[0]))
                            this.trigger(Object.keys(this.players)[1], Environment.actionDecoder(output[1]))

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
            case 0:
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
            case 1:
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
            case 2:
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

    nextStep() {
        Object.keys(this.players).forEach((playerName) => {
            this.players[playerName]["memory"].unshift(
                Environment.getState(this.players[playerName]["actor"], this.players[playerName]["actor"].opponent)
            )
            if (this.players[playerName]["memory"].length > this.memorySize) {
                this.players[playerName]["memory"].pop()
            }

            this.players[playerName]["reward"] += Environment.getReward(this.players[playerName]["actor"])
            this.players[playerName]["reward"] *= 0.5
            console.log(this.players[playerName]["reward"])
        })
    }

    control(playersName) {
        let states = playersName.map((playerName) => {
            return this.players[playerName]["memory"].slice(0, this.ctrlLength)
        })
        let rewards = playersName.map((playerName) => {
            return this.players[playerName]["reward"]
        })
        console.log(states)
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                states: states,
                rewards: rewards,
            }
        })
        console.log("ctrl")
    }

    train(simulationBsz = 1, bsz = 1) {
        this.channel.postMessage({
            instruction: "train",
            args: {
                bsz: bsz
            }
        })
        console.log("train")
    }

    static getState(actorA, actorB) {
        //actorA state
        let getS = (actor) => {
            let HP = actor.HP / actor.maxHP
            let x = (actor.mesh.position.x + 11) / 22
            let y = actor.mesh.position.y / 11
            let faceTo = actor._faceTo == "left" ? 0.1 : 1

            let chapter
            switch (actor._state["chapter"]) {
                case "normal": {
                    chapter = 0.1
                    break
                } case "attack": {
                    chapter = 0.25
                    break
                } case "defense": {
                    chapter = 0.5
                    break
                } case "hitRecover": {
                    chapter = 0.72
                    break
                }
            }

            let section
            switch (actor._state["section"]) {
                case "stand": {
                    section = 0.1
                    break
                } case "jump": {
                    section = 0.25
                    break
                } case "squat": {
                    section = 0.5
                    break
                } case "reStand": {
                    section = 0.75
                    break
                }
            }

            let subsection
            switch (actor._state["subsection"]) {
                case "main": {
                    subsection = 0.1
                    break
                } case "forward": {
                    subsection = 0.25
                    break
                } case "backward": {
                    subsection = 0.5
                    break
                } case "small": {
                    subsection = 1
                    break
                } case "medium": {
                    subsection = 1.25
                    break
                } case "large": {
                    subsection = 1.5
                    break
                } case "fall": {
                    subsection = 2
                    break
                }
            }

            let subsubsection = 0.1 + actor._state["subsubsection"] / 4

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
        let reward = -0.2
        reward += (actor.HP / actor.maxHP) - (actor.opponent.HP / actor.opponent.maxHP)
        if (actor.isPD) {
            reward += 0.5
        }
        if (actor._state["chapter"] == "attack") {
            if (actor.isHit) {
                reward += (actor.opponent.beHitNum * 0.1)
            } else {
                reward -= 0.05
            }
        }
        if (actor._state.chapter == "defense") {
            reward += 0.1
        }
        if (actor.beHitNum != 0) {
            reward -= actor.beHitNum * 0.1
        }

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