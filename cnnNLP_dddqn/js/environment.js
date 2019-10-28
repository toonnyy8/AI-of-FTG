import { Game } from "../../src/lib/slime-FTG/src/js"
import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

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
                action: "null"
            }
            document.addEventListener('keydown', (event) => {
                Object.keys(player["keySet"]).forEach((actionName) => {
                    if (actionName == "attack") {
                        Object.keys(player["keySet"]["attack"]).forEach((actionName) => {
                            if (player["keySet"]["attack"][actionName] == event.key) {
                                last[player["name"]]["action"] = actionName
                            }
                        })
                    } else {
                        if (player["keySet"][actionName] == event.key) {
                            last[player["name"]]["action"] = actionName
                        }
                    }
                })
            })
            document.addEventListener('keyup', (event) => {
                // last[player["name"]]["action"] = "null"
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
                            Object.keys(e.data.args.archive).forEach((playerName) => {
                                this.trigger(playerName, e.data.args.archive[playerName].action)
                            })
                            console.log("ctrl")
                            break
                        }
                    case "train":
                        {
                            this.isReturnTrain = true
                            console.log("train")
                            break
                        }
                    case "save":
                        {
                            tf.tidy(() => {
                                let blob = new Blob([e.data.args.weightsBuffer]);
                                let a = document.createElement("a");
                                let url = window.URL.createObjectURL(blob);
                                let filename = "w.bin";
                                a.href = url;
                                a.download = filename;
                                a.click();
                                window.URL.revokeObjectURL(url);
                            })
                            console.log("save")
                            break
                        }
                    case "load":
                        {
                            alert("load")
                            console.log("load")
                            break
                        }
                    default:
                        break;
                }
            })
        }
    }

    trigger(actorName, action) {
        // // console.log(action)
        switch (action) {
            case 0:
                {
                    break;
                }
            case 1:
                {
                    // // console.log(this.players[actorName].keySet["left"].keydown.key)
                    if (this.players[actorName].actor.shouldFaceTo == "left") {
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keydown
                        )
                    } else {
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keydown
                        )
                    }
                    break;
                }
            case 2:
                {
                    // // console.log(this.players[actorName].keySet["right"].keydown.key)
                    if (this.players[actorName].actor.shouldFaceTo == "left") {
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keydown
                        )
                    } else {
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keyup
                        )
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keydown
                        )
                    }
                    break;
                }
            case 3:
                {
                    // // console.log(this.players[actorName].keySet["jump"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keydown
                    )
                    break;
                }
            case 4:
                {
                    // // console.log(this.players[actorName].keySet["squat"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keydown
                    )
                    break;
                }
            case 5:
                {
                    // // console.log(this.players[actorName].keySet.attack["small"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keydown
                    )
                    break;
                }
            case 6:
                {
                    // // console.log(this.players[actorName].keySet.attack["medium"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keydown
                    )
                    break;
                }
            case 7:
                {
                    // // console.log(this.players[actorName].keySet.attack["large"].keydown.key)
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keydown
                    )
                    break;
                }
            case 8:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["left"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["right"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keyup
                    )
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keyup
                    )
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

            this.players[playerName]["reward"] = this.players[playerName]["reward"] * 0.9 + Environment.getReward(this.players[playerName]["actor"]) * 0.5
            // this.players[playerName]["reward"] *= 0.5
            // console.log(`${playerName} reward : ${this.players[playerName]["reward"]}`)
        })
    }

    control(playerNames, chooseAction) {
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                archive: playerNames.reduce(
                    (acc, playerName) => {
                        acc[playerName] = {
                            state: this.players[playerName]["memory"].slice(0, this.ctrlLength),
                            reward: this.players[playerName]["reward"],
                            action: Environment.getAction(this.players[playerName]["action"])
                        }
                        return acc
                    }, {}),
                chooseAction: chooseAction
            }
        })
        // console.log("ctrl")
    }

    train(bsz = 32, replayIdxes = [null], usePrioritizedReplay = false) {
        this.channel.postMessage({
            instruction: "train",
            args: {
                bsz: bsz,
                replayIdxes: replayIdxes,
                usePrioritizedReplay: usePrioritizedReplay
            }
        })
        // console.log("train")
    }

    init() {
        Object.keys(this.players).forEach((playerName) => {
            this.trigger(playerName, 8)
        })
        Object.values(this.players).forEach((player) => {
            player.memory = new Array(this.memorySize).fill(Environment.getState(player["actor"], player["actor"].opponent))
            player.reward = 0
            player.action = "null"
        })

        console.log("init")
    }
    save() {
        this.channel.postMessage({
            instruction: "save",
            args: {}
        })
        // console.log("save")
    }
    load() {
        tf.tidy(() => {
            let load = document.createElement("input")
            load.type = "file"
            load.accept = ".bin"

            load.onchange = event => {
                const files = load.files
                // console.log(files[0])
                var reader = new FileReader()
                reader.addEventListener("loadend", () => {
                    this.channel.postMessage({
                        instruction: "load",
                        args: {
                            weightsBuffer: new Uint8Array(reader.result)
                        }
                    })
                    // console.log("load")
                });
                reader.readAsArrayBuffer(files[0])
            };

            load.click()
        })
    }

    static getState(actorA, actorB) {
        //actorA state
        let getS = (actor) => {
            let HP = actor.HP / actor.maxHP
            let cumulativeDamage = actor.cumulativeDamage / actor.maxCumulativeDamage
            let x = actor.mesh.position.x / 22
            let y = actor.mesh.position.y / 11
            let faceTo = actor._faceTo == "left" ? -1 : 1

            let chapter
            switch (actor._state["chapter"]) {
                case "normal": {
                    chapter = 1
                    break
                } case "attack": {
                    chapter = 2
                    break
                } case "defense": {
                    chapter = 3
                    break
                } case "hitRecover": {
                    chapter = 4
                    break
                }
            }

            let section
            switch (actor._state["section"]) {
                case "stand": {
                    section = 1
                    break
                } case "jump": {
                    section = 2
                    break
                } case "squat": {
                    section = 3
                    break
                } case "reStand": {
                    section = 4
                    break
                }
            }

            let subsection
            switch (actor._state["subsection"]) {
                case "main": {
                    subsection = 1
                    break
                } case "forward": {
                    subsection = 2
                    break
                } case "backward": {
                    subsection = 3
                    break
                } case "small": {
                    subsection = 4
                    break
                } case "medium": {
                    subsection = 5
                    break
                } case "large": {
                    subsection = 6
                    break
                } case "fall": {
                    subsection = 7
                    break
                }
            }

            let subsubsection = (actor._state["subsubsection"] - 0) + 1

            let frame = actor._state["frame"]

            return [HP, cumulativeDamage, x, y, faceTo, chapter, section, subsection, subsubsection, frame]
        }

        return getS(actorA).concat(getS(actorB))
    }

    static getMask() {
        let getM = () => {
            return [-1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30]
        }

        return getM().concat(getM())
    }

    static getReward(actor) {
        let reward = ((actor.HP / actor.maxHP) - 1) * 0.5
        reward += (actor.HP / actor.maxHP) - (actor.opponent.HP / actor.opponent.maxHP)

        reward -= (actor.cumulativeDamage / actor.maxCumulativeDamage) - (actor.opponent.cumulativeDamage / actor.opponent.maxCumulativeDamage)

        let positionXreward = 0.35 - Math.abs(actor.mesh.position.x - actor.opponent.mesh.position.x) / 22
        reward += Math.min(positionXreward, positionXreward ** 2)

        if (actor.isPD) {
            reward += 1
        }
        if (actor._state["chapter"] == "attack") {
            if (actor.isHit) {
                reward += (actor.opponent.beHitNum * 0.5)
            } else {
                reward -= 0.1
            }
        }
        if (actor._state.chapter == "defense") {
            reward += 0.5
        }
        if (actor.beHitNum != 0) {
            reward -= actor.beHitNum
        }

        return reward
    }

    static actionDecoder(encodeAction) {
        let lr = Math.floor(encodeAction / 12)
        let js = Math.floor((encodeAction - (12 * lr)) / 4)
        let atk = encodeAction - lr * 12 - js * 4
        return [lr, js, atk]
    }

    static getAction(action) {
        switch (action) {
            case "null": {
                return 0
            }
            case "left": {
                return 1
            }
            case "right": {
                return 2
            }
            case "jump": {
                return 3
            }
            case "squat": {
                return 4
            }
            case "small": {
                return 5
            }
            case "medium": {
                return 6
            }
            case "large": {
                return 7
            }
        }
    }

}