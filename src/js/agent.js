import * as tokenSet from "../param/tokens.json"
import * as game from "../lib/slime-FTG/src/js"
import * as transformerXL from "./MirageNet/transformerXL"

export function getStatement(actor = game.getPlayer()[0], actorName = "player1" || "player2", action) {
    return [
        "<info>",
        actorName,
        `hp_${Math.round(actor.HP / 150)}`,
        `faceTo_${actor._faceTo}`,
        `position_x_${Math.round(actor.mesh.position.x / 1.1)}`,
        `position_y_${Math.round(actor.mesh.position.y / 1.1)}`,
        `state_chapter_${actor._state["chapter"]}`,
        `state_section_${actor._state["section"]}`,
        `state_subsection_${actor._state["subsection"]}`,
        `state_subsubsection_${actor._state["subsubsection"]}`,
        `reward_${getReward(actor)}`,
        "</info>",
        "=>",
        "<op>",
        `action_${action["LR"]}`, //none/left/right
        `action_${action["UD"]}`, //none/up/down
        `action_${action["SML"]}`, //none/small/medium/large
        "</op>"
    ].map((word) => {
        // console.log(word)
        return word.split("_").reduce((set, key) => { return set[key] }, tokenSet.tokens)
    })
}

export function getReward(actor = game.getPlayer()[0]) {
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
export function t() {

    let statements = []
    console.log("finished")
    console.log(tokenSet)

    console.log(getStatement())
}

export class Agent {
    constructor(players = [{ name: "player1", actor: game.getPlayer()[0] }]) {
        this.players = players.reduce((last, player) => {
            return last[player["name"]] = { actor: player["actor"], memory: [], action: [] }
        }, {})
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
        })
    }

    control(playerName) {

    }

}