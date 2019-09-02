import "core-js/stable"
import "regenerator-runtime/runtime"
import { Agent } from "./agent"
import { Game } from "../lib/slime-FTG/src/js"
import * as tool from "./tool"

let keySets = [{
    jump: "w",
    squat: "s",
    left: "a",
    right: "d",
    attack: {
        small: "j",
        medium: "k",
        large: "l"
    }
}, {
    jump: "ArrowUp",
    squat: "ArrowDown",
    left: "ArrowLeft",
    right: "ArrowRight",
    attack: {
        small: "1",
        medium: "2",
        large: "3"
    }
}]

let game = new Game(keySets)

let main = () => {
    let agent = new Agent([{
        name: "player1",
        actor: game.player1,
        keySet: keySets[0]
    }, {
        name: "player2",
        actor: game.player2,
        keySet: keySets[1]
    }])

    let mainLoop = new tool.Loop(() => {
        agent.fetchUpReward()
        agent.control("player1", 10)
        agent.control("player2")
        agent.nextStep()
        // console.log(agent.players.player1.memory[agent.players.player1.memory.length - 1])

    }, 6)

    let loop = () => {
        mainLoop.run()
        requestAnimationFrame(loop)
    }
    loop()

}

let checkLoad = () => {
    if (game.player1 && game.player2) {
        main()
    } else {
        setTimeout(checkLoad, 100)
    }
}
setTimeout(checkLoad, 100)