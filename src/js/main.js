import "core-js/stable"
import "regenerator-runtime/runtime"
import * as game from "./game"
// import * as model from "./model/model"

let main = () => {
    console.log("finished")
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