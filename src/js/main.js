import "core-js/stable"
import "regenerator-runtime/runtime"
import "./test"
// import { Agent } from "./agent"
// import { Game } from "../lib/slime-FTG/src/js"
// import * as tool from "./tool"
// import * as tf from "@tensorflow/tfjs"
// import * as tfex from "../lib/tfjs-extensions/src"

// let keySets = [{
//     jump: "w",
//     squat: "s",
//     left: "a",
//     right: "d",
//     attack: {
//         small: "j",
//         medium: "k",
//         large: "l"
//     }
// }, {
//     jump: "ArrowUp",
//     squat: "ArrowDown",
//     left: "ArrowLeft",
//     right: "ArrowRight",
//     attack: {
//         small: "1",
//         medium: "2",
//         large: "3"
//     }
// }]

// let game = new Game(keySets)

// let main = () => {
//     let agent = new Agent([{
//         name: "player1",
//         actor: game.player1,
//         keySet: keySets[0]
//     }, {
//         name: "player2",
//         actor: game.player2,
//         keySet: keySets[1]
//     }])

//     let mainLoop = new tool.Loop(() => {
//         agent.fetchUpReward()
//         // agent.control("player1", 5)
//         // agent.control("player2", 5)
//         agent.nextStep()
//     }, 60)
//     let trainLoop = new tool.Loop(() => {
//         agent.train()
//     }, 3)

//     let loop = () => {
//         mainLoop.run()
//         trainLoop.run()
//         requestAnimationFrame(loop)
//     }
//     loop()
//     agent.control("player1", 5)
//     console.log(tf.memory())
//     agent.control("player2", 5)
//     console.log(tf.memory())
//     console.log(tfex.scope.allVariables())
// }

// let checkLoad = () => {
//     if (game.player1 && game.player2) {
//         main()
//     } else {
//         setTimeout(checkLoad, 100)
//     }
// }
// setTimeout(checkLoad, 100)