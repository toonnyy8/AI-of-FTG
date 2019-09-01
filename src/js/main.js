import "core-js/stable"
import "regenerator-runtime/runtime"
import * as game from "../lib/Slime-FTG/src/js"
import * as tokenSet from "../param/tokens.json"
import * as BABYLON from "babylonjs"
// import * as model from "./model/model"
// import "./test"
let main = () => {
    let statements = []
    console.log("finished")
    console.log(tokenSet)
    let getStatement = (actor = p1, actorName = "player1" || "player2") => {
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
            "reward_-1",
            "</info>",
            "=>",
            "<op>",
            "action_none",
            "</op>"
        ].map((word) => {
            console.log(word)
            return word.split("_").reduce((set, key) => { return set[key] }, tokenSet.tokens)
        })
    }
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

let [p1, p2] = game.getPlayer()
let checkLoad = () => {
    if (p1 && p2) {
        main()
    } else {
        [p1, p2] = game.getPlayer()
        setTimeout(checkLoad, 100)
    }
}
setTimeout(checkLoad, 100)