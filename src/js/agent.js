import * as tokenSet from "../param/tokens.json"
import * as game from "../lib/Slime-FTG/src/js"

export function getStatement(actor = p1, actorName = "player1" || "player2") {
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
        `action_${"none"}`,//none/left/right
        `action_${"none"}`,//none/up/down
        `action_${"none"}`,//none/small/medium/large
        "</op>"
    ].map((word) => {
        console.log(word)
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

export function t() {

    let statements = []
    console.log("finished")
    console.log(tokenSet)

    console.log(getStatement())
    // [tokens["{"]].concat(
    //     statements.reduce((last, statement) => {
    //         if (last == null) {
    //             return statement
    //         } else {
    //             return [tokens["["]].concat(last).concat(tokens["]"]).concat(statement)
    //         }
    //     }, null)
    // ).concat(tokens["}"])
} 