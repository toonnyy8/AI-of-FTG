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
    // let getStatement = (actor = p1,actorName = "player1"||"player2") => {
    //     return [actorName, "<", BABYLON.Vector4. actor.mesh.rotationQuaternion,`${Math.ceil((actor.HP/150))-10}`,"/",">",,"<",">"]
    // }
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