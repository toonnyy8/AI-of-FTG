import "core-js/stable"
import "regenerator-runtime/runtime"
// import "./test"
import { Environment } from "./environment"
import { Game } from "../../src/lib/slime-FTG/src/js"
import * as tool from "../../src/js/tool"
import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

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
    let env = new Environment([{
        name: "player1",
        actor: game.player1,
        keySet: keySets[0]
    }, {
        name: "player2",
        actor: game.player2,
        keySet: keySets[1]
    }], 5000, 1024)

    document.getElementById("player1").onclick = () => {
        if (document.getElementById("player1").innerText == "off") {
            document.getElementById("player1").innerText = "on"
        } else {
            document.getElementById("player1").innerText = "off"
            env.trigger("player1", -1)
        }
    }

    document.getElementById("player2").onclick = () => {
        if (document.getElementById("player2").innerText == "off") {
            document.getElementById("player2").innerText = "on"
        } else {
            document.getElementById("player2").innerText = "off"
            env.trigger("player2", -1)
        }
    }
    document.getElementById("reduceHP").onclick = () => {
        if (document.getElementById("reduceHP").innerText == "off") {
            document.getElementById("reduceHP").innerText = "on"
        } else {
            document.getElementById("reduceHP").innerText = "off"
        }
    }
    document.getElementById("save").onclick = () => {
        env.save()
    }
    document.getElementById("load").onclick = () => {
        env.load()
    }

    let epoch = 100
    let epochCount = epoch

    let ctrlLoop = new tool.Loop(() => {
        if (game.restart) {
            if (env.isReturnCtrl && env.isReturnTrain) {
                if (epochCount == epoch) {
                    env.trigger("player1", -1)
                    env.trigger("player2", -1)
                }
                env.train(1, 2)
                env.isReturnTrain = false
                epochCount -= 1
                if (epochCount == 0) {
                    game.restart = false
                }
            }
        } else {
            epochCount = epoch
            if (env.isReturnTrain) {
                if (document.getElementById("reduceHP").innerText == "on") {
                    game.player1.HP -= 10
                    game.player2.HP -= 10
                }
                env.nextStep()
                if (env.isReturnCtrl) {
                    let players = []
                    if (document.getElementById("player1").innerText == "on") {
                        players.push("player1")
                    }
                    if (document.getElementById("player2").innerText == "on") {
                        players.push("player2")
                    }
                    env.control(players)
                    env.isReturnCtrl = false
                }
            }
        }
    }, 6)

    game.restart = false
    let loop = () => {
        ctrlLoop.run()
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