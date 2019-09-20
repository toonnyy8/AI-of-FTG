import * as tokenSet from "../param/tokens.json"
import { Game } from "../lib/slime-FTG/src/js"
import * as tf from "@tensorflow/tfjs"
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
        chapter = 84
    } else if (chapter == "defense") {
        chapter = 84 * 2
    } else if (chapter == "hitRecover") {
        chapter = 84 * 3
    }

    let section = actor._state["section"]
    if (section == "stand") {
        section = 0
    } else if (section == "jump") {
        section = 28
    } else if (section == "squat") {
        section = 28 * 2
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

    return [player + faceTo + x + y + hp, 1000 + chapter + section + subsection + subsubsection, 1336 + leftOrRight + jumpOrSquat + attack]
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
    }], memorySize = 256) {
        this.memorySize = memorySize
        this.players = players.reduce((last, player) => {
            last[player["name"]] = {
                actor: player["actor"],
                keySet: player["keySet"],
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
            switch (e.data.instruction) {
                case "ctrl": {
                    this.isReturnCtrl = true
                    let p1a = e.data.output[0].pop().slice(1336, 1336 + 36).reduce((ret, curr, idx) => {
                        if (ret.val < curr) {
                            ret.idx = idx
                            ret.val = curr
                        }
                        return ret
                    }, { val: 0, idx: 0 })
                    let p2a = e.data.output[1].pop().slice(1336, 1336 + 36).reduce((ret, curr, idx) => {
                        if (ret.val < curr) {
                            ret.idx = idx
                            ret.val = curr
                        }
                        return ret
                    }, { val: 0, idx: 0 })
                    console.log(p1a, p2a)
                    break
                }
                case "train": {
                    this.isReturnTrain = true
                    break
                }
                default:
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

    control(playersName, expectedReward = 0) {
        let inps = playersName.map((playerName) => {
            let pReward = Math.min(Math.round(this.predictReward(playerName) + 1), 5)
            let newStatement = getStatement(this.players[playerName]["actor"], playerName, this.players[playerName]["action"])
            newStatement[2] = 1372 + 6 + pReward

            let [inp, _] = this.mergeMemory(playerName, 5)
            inp.push(newStatement)
            return [inp.flat()]
        })
        console.log(inps)
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                inps: inps,
                tgts: inps,
                nToken: 1384,
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

    train() {
        let tgts = []
        let rewards = []
        for (let i = 0; i < 2; i++) {
            tgts = tgts.concat(
                Object.keys(this.players).map((playerName) => {
                    let end = Math.round(Math.random() * (this.memorySize - 6) + 5)
                    let [tgt, reward] = this.mergeMemory(playerName, 5, end)
                    rewards.push(reward)
                    return [tgt.flat()]
                })
            )
        }
        let inps = tgts.map((tgt, tgtIdx) => {
            return [tgt.map((words) => {
                let inp = []
                for (let i = 0; i < words.length; i++) {
                    if (i % 3 == 2) {
                        if (Math.random() > 0.5) {
                            inp.push(rewards[tgtIdx][Math.floor(i / 3)])
                        } else {
                            inp.push(words[i])
                        }
                    } else {
                        inp.push(words[i])
                    }
                }
                return inp
            }).flat()]
        })
        console.log(inps)
        this.channel.postMessage({
            instruction: "train",
            args: {
                inps: inps,
                tgts: tgts,
                nToken: 1384,
                FLAGS: FLAGS
            }
        })
        console.log("train")
    }
}