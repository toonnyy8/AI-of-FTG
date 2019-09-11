import * as tokenSet from "../param/tokens.json"
import { Game } from "../lib/slime-FTG/src/js"
import * as tf from "@tensorflow/tfjs"
import * as FLAGS from "../param/flags.json"

tf.setBackend("webgl")
// tf.enableProdMode()

export function getStatement(actor, actorName = "player1" || "player2", action) {
    let stateVector = []
    stateVector[0] = actorName == "player1" ? 1 : 0
    stateVector[1] = actorName == "player2" ? 1 : 0
    stateVector[2] = actor.HP / 3000
    stateVector[3] = actor._faceTo == "left" ? 1 : 0
    stateVector[4] = actor._faceTo == "left" ? 1 : 0
    stateVector[5] = (actor.mesh.position.x + 11) / 22
    stateVector[6] = actor.mesh.position.y / 11
    stateVector[7] = actor._state["chapter"] == "normal" ? 1 : 0
    stateVector[8] = actor._state["chapter"] == "attack" ? 1 : 0
    stateVector[9] = actor._state["chapter"] == "defense" ? 1 : 0
    stateVector[10] = actor._state["chapter"] == "hitRecover" ? 1 : 0
    stateVector[11] = actor._state["section"] == "stand" ? 1 : 0
    stateVector[12] = actor._state["section"] == "jump" ? 1 : 0
    stateVector[13] = actor._state["section"] == "squat" ? 1 : 0
    stateVector[14] = actor._state["subsection"] == "main" ? 1 : 0
    stateVector[15] = actor._state["subsection"] == "forward" ? 1 : 0
    stateVector[16] = actor._state["subsection"] == "backward" ? 1 : 0
    stateVector[17] = actor._state["subsection"] == "small" ? 1 : 0
    stateVector[18] = actor._state["subsection"] == "medium" ? 1 : 0
    stateVector[19] = actor._state["subsection"] == "large" ? 1 : 0
    stateVector[20] = actor._state["subsection"] == "fall" ? 1 : 0
    stateVector[21] = actor._state["subsubsection"] == "0" ? 1 : 0
    stateVector[22] = actor._state["subsubsection"] == "1" ? 1 : 0
    stateVector[23] = actor._state["subsubsection"] == "2" ? 1 : 0
    stateVector[24] = actor._state["subsubsection"] == "3" ? 1 : 0
    stateVector[25] = action["left"]
    stateVector[26] = action["right"]
    stateVector[27] = action["jump"]
    stateVector[28] = action["squat"]
    stateVector[29] = action["small"]
    stateVector[30] = action["medium"]
    stateVector[31] = action["large"]
    stateVector[32] = 0//reward
    return stateVector
    return [
        "<info>",
        actorName,
        // `hp_${Math.round(actor.HP / 150)}`,
        // `faceTo_${actor._faceTo}`,
        // `position_x_${Math.round(actor.mesh.position.x / 1.1)}`,
        // `position_y_${Math.round(actor.mesh.position.y / 1.1)}`,
        // `state_chapter_${actor._state["chapter"]}`,
        // `state_section_${actor._state["section"]}`,
        // `state_subsection_${actor._state["subsection"]}`,
        // `state_subsubsection_${actor._state["subsubsection"]}`,
        // "</info>",
        // "<op>",
        // `action_${action["left"] ? "left" : "none"}`, //none/left
        // `action_${action["right"] ? "right" : "none"}`, //none/right
        // `action_${action["jump"] ? "jump" : "none"}`, //none/jump
        // `action_${action["squat"] ? "squat" : "none"}`, //none/squat
        // `action_${action["small"] ? "small" : "none"}`, //none/small
        // `action_${action["medium"] ? "medium" : "none"}`, //none/medium
        // `action_${action["large"] ? "large" : "none"}`, //none/large
        // "</op>",
        // "=>",
        // "<reward>",
        // `mask`,
        "</reward>"
    ].map((word) => {
        // console.log(word)
        return word.split("_").reduce((set, key) => { return set[key] }, tokenSet.tokens)
    })
}

export function getReward(actor) {
    let reward = Math.round((actor.HP - actor.opponent.HP) / 1500)
    if (actor.isPD) {
        reward += 10
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

    reward = Math.min(Math.max(reward, -10), 10)

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
    }], memorySize = 256) {
        this.memorySize = memorySize
        this.players = players.reduce((last, player) => {
            last[player["name"]] = {
                actor: player["actor"],
                keySet: player["keySet"],
                memory: [],
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
        this.isReturn = true
        this.channel.onmessage = (e) => {
            this.isReturn = true
            console.log(this.isReturn)
        }
    }

    fetchUpReward() {
        Object.keys(this.players).forEach((playerName) => {
            if (this.players[playerName]["memory"].length != 0) {
                let lastStatement = this.players[playerName]["memory"].pop()
                lastStatement[lastStatement.indexOf(tokenSet.tokens["<reward>"]) + 1] = tokenSet.tokens["reward"][`${getReward(this.players[playerName]["actor"])}`]
                this.players[playerName]["memory"].push(lastStatement)
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

    control(playersName, expectedReward = 0) {
        tf.tidy(() => {
            let inps = playersName.map((playerName) => {
                let newStatement = getStatement(this.players[playerName]["actor"], playerName, this.players[playerName]["action"])
                newStatement = maskAction(newStatement)
                newStatement[newStatement.indexOf(tokenSet.tokens["<reward>"]) + 1] = tokenSet.tokens["reward"][`${expectedReward}`]

                let inp = this.mergeMemory(playerName, 5)
                inp.push(newStatement)
                return [inp.flat()]
            })
            console.log(inps)
            this.channel.postMessage({
                instruction: "ctrl",
                args: {
                    inps: inps,
                    tgts: inps,
                    nToken: tokenSet.nToken,
                    FLAGS: FLAGS
                }
            })
            console.log("ctrl")
        })
    }

    mergeMemory(mainPlayerName, mergeLength, end = this.players[mainPlayerName]["memory"].length - 1) {
        let mergeMem = []
        end = end <= this.players[mainPlayerName]["memory"].length - 1 ? end : this.players[mainPlayerName]["memory"].length - 1
        mergeLength = end - mergeLength >= 0 ? mergeLength : end
        for (let i = end - mergeLength + 1; i <= end; i++) {
            mergeMem.push(this.players[mainPlayerName]["memory"][i].slice())
            Object.keys(this.players).forEach((playerName) => {
                if (playerName != mainPlayerName) {
                    mergeMem.push(this.players[playerName]["memory"][i])
                }
            })
        }
        return mergeMem
    }

    setAction(playerName, action) {
        this.players[playerName]["actor"] = action
    }

    train() {

    }
}