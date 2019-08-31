import "core-js/stable"
import "regenerator-runtime/runtime"
import * as game from "./game"
import * as tokens from "../param/tokens.json"
// import * as model from "./model/model"
// import "./test"

let main = () => {
    console.log("finished")
    console.log(tokens)
    let getStatement = (actor = p1) => {
    }
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