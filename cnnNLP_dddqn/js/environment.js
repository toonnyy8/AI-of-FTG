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
        canvas,
        memorySize = 256,
        ctrlLength = 5
    ) {
        this.memorySize = memorySize
        this.ctrlLength = ctrlLength
        this.canvas = canvas
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
                memory: new Array(this.memorySize),
                // reward: 0,
                point: 0,
                keyDown: Object.keys(player["keySet"])
                    .reduce((last, actionName) => {
                        if (actionName == "attack") {
                            last[actionName] = Object.keys(player["keySet"]["attack"])
                                .reduce((last, attackName) => {
                                    last[attackName] = false
                                    return last
                                }, {})
                        } else {
                            last[actionName] = false
                        }
                        return last
                    }, {})
            }
            last[player["name"]]["memory"].fill(Environment.getState(last[player["name"]]))
            document.addEventListener('keydown', (event) => {
                Object.keys(player["keySet"]).forEach((actionName) => {
                    if (actionName == "attack") {
                        Object.keys(player["keySet"]["attack"]).forEach((actionName) => {
                            if (player["keySet"]["attack"][actionName] == event.key) {
                                last[player["name"]]["keyDown"]["attack"][actionName] = true
                            }
                        })
                    } else {
                        if (player["keySet"][actionName] == event.key) {
                            last[player["name"]]["keyDown"][actionName] = true
                        }
                    }
                })
            })
            document.addEventListener('keyup', (event) => {
                // last[player["name"]]["action"] = "null"
                Object.keys(player["keySet"]).forEach((actionName) => {
                    if (actionName == "attack") {
                        Object.keys(player["keySet"]["attack"]).forEach((actionName) => {
                            if (player["keySet"]["attack"][actionName] == event.key) {
                                last[player["name"]]["keyDown"]["attack"][actionName] = false
                            }
                        })
                    } else {
                        if (player["keySet"][actionName] == event.key) {
                            last[player["name"]]["keyDown"][actionName] = false
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
                            Object.keys(e.data.args.archive).forEach((playerName) => {
                                this.trigger(playerName, e.data.args.archive[playerName].actions)
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

    // trigger(actorName, actions) {
    //     // console.log(actions)

    //     switch (actions[0]) {
    //         case 0:
    //             {
    //                 break;
    //             }
    //         case 1:
    //             {
    //                 if (this.players[actorName].keyDown.left) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 if (this.players[actorName].keyDown.right) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //     }
    //     switch (actions[1]) {
    //         case 0:
    //             {
    //                 break;
    //             }
    //         case 1:
    //             {
    //                 if (this.players[actorName].keyDown.jump) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["jump"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["jump"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 if (this.players[actorName].keyDown.squat) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["squat"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["squat"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //     }
    //     switch (actions[2]) {
    //         case 0:
    //             {
    //                 break;
    //             }
    //         case 1:
    //             {
    //                 if (this.players[actorName].keyDown.attack.small) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["small"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["small"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 if (this.players[actorName].keyDown.attack.medium) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["medium"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["medium"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 3:
    //             {
    //                 if (this.players[actorName].keyDown.attack.large) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["large"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["large"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //     }
    // }
    // trigger(actorName, actions) {
    //     // console.log(actions)

    //     switch (actions[0]) {
    //         case 0:
    //             {
    //                 break;
    //             }
    //         case 1:
    //             {
    //                 if (this.players[actorName].keyDown.left) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 if (this.players[actorName].keyDown.right) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 3:
    //             {
    //                 if (this.players[actorName].keyDown.jump) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["jump"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["jump"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 4:
    //             {
    //                 if (this.players[actorName].keyDown.squat) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["squat"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["squat"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 5:
    //             {
    //                 if (this.players[actorName].keyDown.attack.small) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["small"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["small"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 6:
    //             {
    //                 if (this.players[actorName].keyDown.attack.medium) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["medium"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["medium"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 7:
    //             {
    //                 if (this.players[actorName].keyDown.attack.large) {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["large"].keyup
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet.attack["large"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //     }
    // }
    trigger(actorName, actions) {
        // console.log(actions)

        switch (actions[0]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["left"].keydown
                    )
                    break;
                }
            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["left"].keyup
                    )
                    break;
                }
        }
        switch (actions[1]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["right"].keydown
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["right"].keyup
                    )
                    break;
                }
        }
        switch (actions[2]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keydown
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keyup
                    )
                    break;
                }

        }
        switch (actions[3]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keydown
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keyup
                    )
                    break;
                }

        }
        switch (actions[4]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keydown
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keyup
                    )
                    break;
                }

        }
        switch (actions[5]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keydown
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keyup
                    )
                    break;
                }

        }
        switch (actions[6]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keydown
                    )
                    break;
                }

            case 1:
                {
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
                Environment.getState(this.players[playerName])
            )
            if (this.players[playerName]["memory"].length > this.memorySize) {
                this.players[playerName]["memory"].pop()
            }

            this.players[playerName]["point"] = [
                Environment.getPoint(this.players[playerName]["actor"]),
                Environment.getPoint(this.players[playerName]["actor"]),
                Environment.getPoint(this.players[playerName]["actor"]),
                Environment.getPoint(this.players[playerName]["actor"]),
                Environment.getPoint(this.players[playerName]["actor"]),
                Environment.getPoint(this.players[playerName]["actor"]),
                Environment.getPoint(this.players[playerName]["actor"])
                // Environment.getMovePoint(this.players[playerName]["actor"]),
                // Environment.getJumpPoint(this.players[playerName]["actor"]),
                // Environment.getAttackPoint(this.players[playerName]["actor"])
            ]
            // console.log(`${playerName} reward : ${Math.round(this.players[playerName]["point"] * 10000) / 10000}`)
        })
    }

    control(playerNames, chooseActionRandomValue) {
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                archive: playerNames.reduce(
                    (acc, playerName) => {
                        acc[playerName] = {
                            state: this.players[playerName]["memory"].slice(0, this.ctrlLength),
                            rewards: this.players[playerName]["point"]
                        }
                        return acc
                    }, {}),
                chooseActionRandomValue: chooseActionRandomValue
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
            player.memory = new Array(this.memorySize).fill(Environment.getState(player))
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

    static getState(player) {
        //actorA state
        let getActorState = (actor) => {
            let HP = actor.HP / actor.maxHP
            let cumulativeDamage = actor.cumulativeDamage / actor.maxCumulativeDamage
            let x = actor.mesh.position.x / 22
            let y = actor.mesh.position.y / 11
            let faceTo = actor._faceTo == "left" ? -1 : 1

            let chapter = new Array(4).fill(-1 * actor._state["frame"])
            switch (actor._state["chapter"]) {
                case "normal": {
                    chapter[0] = actor._state["frame"]
                    break
                } case "attack": {
                    chapter[1] = actor._state["frame"]
                    break
                } case "defense": {
                    chapter[2] = actor._state["frame"]
                    break
                } case "hitRecover": {
                    chapter[3] = actor._state["frame"]
                    break
                }
            }

            let section = new Array(4).fill(-1 * actor._state["frame"])
            switch (actor._state["section"]) {
                case "stand": {
                    section[0] = actor._state["frame"]
                    break
                } case "jump": {
                    section[1] = actor._state["frame"]
                    break
                } case "squat": {
                    section[2] = actor._state["frame"]
                    break
                } case "reStand": {
                    section[3] = actor._state["frame"]
                    break
                }
            }

            let subsection = new Array(7).fill(-1 * actor._state["frame"])
            switch (actor._state["subsection"]) {
                case "main": {
                    subsection[0] = actor._state["frame"]
                    break
                } case "forward": {
                    subsection[1] = actor._state["frame"]
                    break
                } case "backward": {
                    subsection[2] = actor._state["frame"]
                    break
                } case "small": {
                    subsection[3] = actor._state["frame"]
                    break
                } case "medium": {
                    subsection[4] = actor._state["frame"]
                    break
                } case "large": {
                    subsection[5] = actor._state["frame"]
                    break
                } case "fall": {
                    subsection[6] = actor._state["frame"]
                    break
                }
            }
            let subsubsection = new Array(4).fill(-1 * actor._state["frame"])
            subsubsection[actor._state["subsubsection"]] = actor._state["frame"]

            return [HP, cumulativeDamage, x, y, faceTo]
                .concat(chapter)
                .concat(section)
                .concat(subsection)
                .concat(subsubsection)
        }

        return getActorState(player["actor"])
            .concat(getActorState(player["actor"].opponent))
            .concat(Object.values(player["keyDown"]).reduce((last, v) => {
                if (Object.values(v).length != 0) {
                    return last.concat(Object.values(v))
                } else {
                    return last.concat(v)
                }
            }, [])
                .map((v) => {
                    let faceTo = player["actor"]._faceTo == "left" ? -1 : 1
                    return v ?
                        faceTo :
                        -1 * faceTo
                })
            )
    }

    static getMask() {
        let getM = () => {
            return [-1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30]
        }

        return getM().concat(getM())
    }

    static getPoint(actor) {
        let point = (actor.HP / actor.maxHP) - 1
        point += (actor.HP / actor.maxHP) - (actor.opponent.HP / actor.opponent.maxHP)

        point -= (actor.cumulativeDamage / actor.maxCumulativeDamage)
        // point += (actor.opponent.cumulativeDamage / actor.opponent.maxCumulativeDamage)

        point *= 1 - (Math.abs(actor.mesh.position.x - actor.opponent.mesh.position.x) / 22)

        if (actor._state["chapter"] == "attack") {
            if (actor.isHit) {
                point += (actor.opponent.beHitNum * Math.abs(point))
            } else {
                point -= Math.abs(point)
            }
        }
        if (actor.beHitNum != 0) {
            point -= actor.beHitNum * Math.abs(point)
        } else {
            if (actor.opponent._state["chapter"] == "attack") {
                point += Math.abs(point) * 2
            }
        }
        if (actor.isPD) {
            point = Math.abs(point)
        }
        if (actor._state.chapter == "defense") {
            point *= 0.5
        }

        return point
    }
    static getMovePoint(actor) {
        let point = (actor.HP / actor.maxHP) - 1
        point += (actor.HP / actor.maxHP) - (actor.opponent.HP / actor.opponent.maxHP)

        point -= (actor.cumulativeDamage / actor.maxCumulativeDamage)
        // point += (actor.opponent.cumulativeDamage / actor.opponent.maxCumulativeDamage)

        point *= 1 - (Math.abs(actor.mesh.position.x - actor.opponent.mesh.position.x) / 22)

        if (actor.isPD) {
            point = Math.abs(point)
        }
        if (actor._state.chapter == "defense") {
            point *= 0.5
        }
        if (actor.beHitNum != 0) {
            point -= actor.beHitNum * Math.abs(point)
        }

        return point
    }

    static getJumpPoint(actor) {
        let point = (actor.HP / actor.maxHP) - 1
        point += (actor.HP / actor.maxHP) - (actor.opponent.HP / actor.opponent.maxHP)

        point -= (actor.cumulativeDamage / actor.maxCumulativeDamage)
        // point += (actor.opponent.cumulativeDamage / actor.opponent.maxCumulativeDamage)

        point *= 1 - (Math.abs(actor.mesh.position.y - actor.opponent.mesh.position.y) / 10)

        return point
    }

    static getAttackPoint(actor) {
        let point = -1 * ((actor.cumulativeDamage / actor.maxCumulativeDamage) - (actor.opponent.cumulativeDamage / actor.opponent.maxCumulativeDamage))

        if (actor._state["chapter"] == "attack") {
            if (actor.isHit) {
                point += (actor.opponent.beHitNum * Math.abs(point))
            } else {
                point -= Math.abs(point)
            }
        }
        if (actor.beHitNum != 0) {
            point -= actor.beHitNum * Math.abs(point)
        }

        return point
    }
}