import "core-js/stable"
import "regenerator-runtime/runtime"
import * as game from "./game"
import * as tokens from "../param/tokens.json"
import * as BABYLON from "babylonjs"
// import * as model from "./model/model"
// import "./test"
let main = () => {
    let statements = []
    console.log("finished")
    console.log(Object.keys(tokens.default))
    console.log(tokens.default)
    let getStatement = (actor = p1, actorName = "player1" || "player2") => {
        return [
            "<start>",
            "player", "_", "1",
            "/",
            "hp", "_", "+", "3", "0", "0", "0",
            "/",
            "faceTo", "_", "+", "1",
            "/",
            "chapter", "_", "0",
            "/",
            "section", "_", "0",
            "/",
            "subsection", "_", "0",
            "/",
            "subsubsection", "_", "0",
            "/",
            "reward", "_", "+", "0", "1",
            "=>",
            "up", "_", "0",
            "/",
            "down", "_", "0",
            "/",
            "left", "_", "0",
            "/",
            "right", "_", "0",
            "/",
            "small", "_", "0",
            "/",
            "medium", "_", "0",
            "/",
            "large", "_", "0",
            "<end>"
        ].map((word) => {
            return tokens[word]
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