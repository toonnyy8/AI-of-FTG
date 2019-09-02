import "core-js/stable"
import "regenerator-runtime/runtime"
import * as agent from "./agent"
import * as game from "../lib/slime-FTG/src/js"
import * as tool from "./tool"

let mainLoop = new tool.Loop(() => {
        console.log(
            agent.maskAction(
                agent.getStatement(p1)
            )
        )
    }, 6)
    // import "./test"
let main = () => {
    requestAnimationFrame(main)
    mainLoop.run()
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