import "core-js/stable"
import "regenerator-runtime/runtime"
import * as agent from "./agent"
import * as game from "../lib/Slime-FTG/src/js"

// import "./test"
let main = () => {
    requestAnimationFrame(main)
    console.log(agent.getStatement(p1))
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