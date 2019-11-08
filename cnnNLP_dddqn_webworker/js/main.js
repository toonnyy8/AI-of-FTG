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

let canvas = document.getElementById("bobylonCanvas")

let game = new Game(keySets, canvas)

let main = () => {
    // game.player1.maxHP = 1000
    // game.player2.maxHP = 1000

    let env = new Environment([{
        name: "player1",
        actor: game.player1,
        keySet: keySets[0]
    }, {
        name: "player2",
        actor: game.player2,
        keySet: keySets[1]
    }], canvas, 5000, 16)

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
    document.getElementById("trainAtFrame").onclick = () => {
        if (document.getElementById("trainAtFrame").innerText == "off") {
            document.getElementById("trainAtFrame").innerText = "on"
        } else {
            document.getElementById("trainAtFrame").innerText = "off"
        }
    }
    document.getElementById("trainAtEnd").onclick = () => {
        if (document.getElementById("trainAtEnd").innerText == "off") {
            document.getElementById("trainAtEnd").innerText = "on"
        } else {
            document.getElementById("trainAtEnd").innerText = "off"
        }
    }
    document.getElementById("save").onclick = () => {
        env.save()
    }
    document.getElementById("load").onclick = () => {
        env.load()
    }

    let maxEpoch = 100
    let epochCount = maxEpoch
    let ctrlNum = 0

    let trainLoop = new tool.Loop(() => {
        if (document.getElementById("trainAtFrame").innerText == "on") {
            env.train(64, [], false)
        }
    }, 32)

    let ctrlLoop = new tool.Loop(() => {
        if (env.isReturnCtrl) {
            let players = []
            let chooseActionRandomValue = []
            if (document.getElementById("player1").innerText == "on") {
                players.push("player1")
                chooseActionRandomValue.push(document.getElementById("player1ChooseActionRandomValue").value)
            }
            if (document.getElementById("player2").innerText == "on") {
                players.push("player2")
                chooseActionRandomValue.push(document.getElementById("player2ChooseActionRandomValue").value)
            }
            env.control(players, chooseActionRandomValue)
            trainLoop.run()

            env.isReturnCtrl = false

            ctrlNum += 1
        }
    }, document.getElementById("controlPeriod").value)

    document.getElementById("controlPeriod").onchange = () => {
        ctrlLoop.period = document.getElementById("controlPeriod").value
    }

    game.restart = false
    let loop = () => {
        if (game.restart) {
            if (document.getElementById("trainAtEnd").innerText == "off") {
                game.restart = false
                env.init()
            } else if (env.isReturnCtrl && env.isReturnTrain) {
                env.train(64, [], true)
                env.isReturnTrain = false
                epochCount -= 1
                if (epochCount == 0) {
                    ctrlNum = 0
                    game.restart = false
                    env.init()
                }
            }
        } else {
            epochCount = Math.min(maxEpoch, Math.ceil(ctrlNum * 2 / 32))
            // console.log(epochCount)
            if (env.isReturnTrain) {
                if (document.getElementById("reduceHP").innerText == "on") {
                    game.player1.HP -= 0.5
                    game.player2.HP -= 0.5
                }
                // console.clear()
                env.nextStep()
                ctrlLoop.run()
            }
        }
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