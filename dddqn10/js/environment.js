import { Game } from "../../src/lib/slime-FTG/src/js"
import * as tf from "@tensorflow/tfjs"
import { registerTfex } from "../../src/lib/tfjs-extensions/src"
const tfex = registerTfex(tf)

tf.setBackend("webgl")
// tf.setBackend("cpu")
// tf.enableProdMode()

export class Environment {
    constructor(
        players = [{
            name: "player1",
            actor: new Game().player1,
            keySet: {
                jump: "w",
                squat: "s",
                left: "a",
                right: "d",
                attack: {
                    small: "j",
                    medium: "k",
                    large: "l"
                }
            }
        }],
        canvas,
        memorySize = 256,
        ctrlLength = 5
    ) {
        this.memorySize = memorySize
        this.ctrlLength = ctrlLength
        this.canvas = canvas
        this.players = players.reduce((last, player) => {
            last[player["name"]] = {
                actor: player["actor"],
                keySet: Object.keys(player["keySet"])
                    .reduce((last, actionName) => {
                        if (actionName == "attack") {
                            last[actionName] = Object.keys(player["keySet"]["attack"])
                                .reduce((last, attackName) => {
                                    last[attackName] = {
                                        keyup: new KeyboardEvent("keyup", {
                                            key: player["keySet"]["attack"][attackName]
                                        }),
                                        keydown: new KeyboardEvent("keydown", {
                                            key: player["keySet"]["attack"][attackName]
                                        })
                                    }
                                    return last
                                }, {})
                        } else {
                            last[actionName] = {
                                keyup: new KeyboardEvent("keyup", {
                                    key: player["keySet"][actionName]
                                }),
                                keydown: new KeyboardEvent("keydown", {
                                    key: player["keySet"][actionName]
                                })
                            }
                        }
                        return last
                    }, {}),
                memory: new Array(this.memorySize),
                // reward: 0,
                point: 0
            }
            last[player["name"]]["memory"].fill(Environment.getState(last[player["name"]]))
            return last
        }, {})
        this.steps = 0

        this.channel = new BroadcastChannel('dddqn10');
        this.isReturnCtrl = true
        this.isReturnTrain = true
        this.channel.onmessage = (e) => {
            tf.tidy(() => {
                switch (e.data.instruction) {
                    case "ctrl":
                        {
                            this.isReturnCtrl = true
                            Object.keys(e.data.args.archives).forEach((playerName) => {
                                // let actions = e.data.args.archives[playerName].actions
                                let actions = []
                                let emb = e.data.args.archives[playerName].actions[0]
                                for (let i = 0; i < 7; i++) {
                                    actions[i] = emb % 2
                                    emb = Math.floor(emb / 2)
                                }
                                if (e.data.args.archives[playerName].aiCtrl) {
                                    this.trigger(playerName, actions)
                                }
                            })
                            console.log("ctrl")
                            break
                        }
                    case "train":
                        {
                            this.isReturnTrain = true
                            console.log("train")
                            break
                        }
                    case "save":
                        {
                            tf.tidy(() => {
                                let blob = new Blob([e.data.args.weightsBuffer]);
                                let a = document.createElement("a");
                                let url = window.URL.createObjectURL(blob);
                                let filename = "w.bin";
                                a.href = url;
                                a.download = filename;
                                a.click();
                                window.URL.revokeObjectURL(url);
                            })
                            console.log("save")
                            break
                        }
                    case "load":
                        {
                            alert("load")
                            console.log("load")
                            break
                        }
                    case "updatePrioritys":
                        {
                            this.isReturnTrain = true
                            console.log("updatePrioritys")
                            break
                        }
                    default:
                        break;
                }
            })
        }
    }

    // trigger(actorName, actions) {
    //     // console.log(actions)

    //     switch (actions[0]) {
    //         case 0:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["jump"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["squat"].keyup
    //                 )
    //                 break;
    //             }
    //         case 1:
    //             {

    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["squat"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["jump"].keydown
    //                 )
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["jump"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["squat"].keydown
    //                 )
    //                 break;
    //             }
    //     }
    //     switch (actions[1]) {
    //         case 0:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["left"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet["right"].keyup
    //                 )
    //                 break;
    //             }
    //         case 1:
    //             {
    //                 if (this.players[actorName].actor.shouldFaceTo == "left") {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keyup
    //                     )
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keydown
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keyup
    //                     )
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 if (this.players[actorName].actor.shouldFaceTo == "right") {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keyup
    //                     )
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keydown
    //                     )
    //                 } else {
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["left"].keyup
    //                     )
    //                     document.dispatchEvent(
    //                         this.players[actorName].keySet["right"].keydown
    //                     )
    //                 }
    //                 break;
    //             }
    //     }
    //     switch (actions[2]) {
    //         case 0:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["small"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["medium"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["large"].keyup
    //                 )
    //                 break;
    //             }
    //         case 1:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["medium"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["large"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["small"].keydown
    //                 )
    //                 break;
    //             }
    //         case 2:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["small"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["large"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["medium"].keydown
    //                 )
    //                 break;
    //             }
    //         case 3:
    //             {
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["small"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["medium"].keyup
    //                 )
    //                 document.dispatchEvent(
    //                     this.players[actorName].keySet.attack["large"].keydown
    //                 )
    //                 break;
    //             }
    //     }
    //     // switch (actions[3]) {
    //     //     case 0:
    //     //         {
    //     //             document.dispatchEvent(
    //     //                 this.players[actorName].keySet.attack["small"].keyup
    //     //             )
    //     //             break;
    //     //         }
    //     //     case 1:
    //     //         {
    //     //             if (this.players[actorName].actor.keyDown.attack.small && Math.random() > 0.8) {
    //     //                 document.dispatchEvent(
    //     //                     this.players[actorName].keySet.attack["small"].keyup
    //     //                 )
    //     //             } else {
    //     //                 document.dispatchEvent(
    //     //                     this.players[actorName].keySet.attack["small"].keydown
    //     //                 )
    //     //             }
    //     //             break;
    //     //         }
    //     // }
    //     // switch (actions[4]) {
    //     //     case 0:
    //     //         {
    //     //             document.dispatchEvent(
    //     //                 this.players[actorName].keySet.attack["medium"].keyup
    //     //             )
    //     //             break;
    //     //         }
    //     //     case 1:
    //     //         {
    //     //             if (this.players[actorName].actor.keyDown.attack.medium && Math.random() > 0.8) {
    //     //                 document.dispatchEvent(
    //     //                     this.players[actorName].keySet.attack["medium"].keyup
    //     //                 )
    //     //             } else {
    //     //                 document.dispatchEvent(
    //     //                     this.players[actorName].keySet.attack["medium"].keydown
    //     //                 )
    //     //             }
    //     //             break;
    //     //         }
    //     // }
    //     // switch (actions[5]) {
    //     //     case 0:
    //     //         {
    //     //             document.dispatchEvent(
    //     //                 this.players[actorName].keySet.attack["large"].keyup
    //     //             )
    //     //             break;
    //     //         }
    //     //     case 1:
    //     //         {
    //     //             if (this.players[actorName].actor.keyDown.attack.large && Math.random() > 0.8) {
    //     //                 document.dispatchEvent(
    //     //                     this.players[actorName].keySet.attack["large"].keyup
    //     //                 )
    //     //             } else {
    //     //                 document.dispatchEvent(
    //     //                     this.players[actorName].keySet.attack["large"].keydown
    //     //                 )
    //     //             }
    //     //             break;
    //     //         }
    //     // }
    // }
    trigger(actorName, actions) {
        // console.log(actions)

        switch (actions[0]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keyup
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["jump"].keydown
                    )
                    break;
                }

        }
        switch (actions[1]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keyup
                    )
                    break;
                }

            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet["squat"].keydown
                    )
                    break;
                }

        }
        switch (actions[2]) {
            case 0:
                {
                    if (this.players[actorName].actor.shouldFaceTo == "left") {
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keyup
                        )
                    } else {
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keyup
                        )
                    }
                    break;
                }
            case 1:
                {
                    if (this.players[actorName].actor.shouldFaceTo == "left") {
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keydown
                        )
                    } else {
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keydown
                        )
                    }
                    break;
                }
        }
        switch (actions[3]) {
            case 0:
                {
                    if (this.players[actorName].actor.shouldFaceTo == "right") {
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keyup
                        )
                    } else {
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keyup
                        )
                    }
                    break;
                }
            case 1:
                {
                    if (this.players[actorName].actor.shouldFaceTo == "right") {
                        document.dispatchEvent(
                            this.players[actorName].keySet["left"].keydown
                        )
                    } else {
                        document.dispatchEvent(
                            this.players[actorName].keySet["right"].keydown
                        )
                    }
                    break;
                }
        }
        switch (actions[4]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keyup
                    )
                    break;
                }
            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["small"].keydown
                    )
                    break;
                }
        }
        switch (actions[5]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keyup
                    )
                    break;
                }
            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["medium"].keydown
                    )
                    break;
                }
        }
        switch (actions[6]) {
            case 0:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keyup
                    )
                    break;
                }
            case 1:
                {
                    document.dispatchEvent(
                        this.players[actorName].keySet.attack["large"].keydown
                    )
                    break;
                }
        }
    }

    nextStep() {
        this.steps += 1
        Object.keys(this.players).forEach((playerName) => {
            this.players[playerName]["memory"].unshift(
                Environment.getState(this.players[playerName])
            )
            if (this.players[playerName]["memory"].length > this.memorySize) {
                this.players[playerName]["memory"].pop()
            }

            if (this.players[playerName]["point"].length !== undefined) {
                this.players[playerName]["point"] = this.players[playerName]["point"].map(point => {
                    return point + Environment.getPoint(this.players[playerName]["actor"])
                })
            } else {
                this.players[playerName]["point"] = [
                    Environment.getPoint(this.players[playerName]["actor"]),
                    // Environment.getPoint(this.players[playerName]["actor"]),
                    // Environment.getPoint(this.players[playerName]["actor"]),
                    // Environment.getPoint(this.players[playerName]["actor"]),
                    // Environment.getPoint(this.players[playerName]["actor"]),
                    // Environment.getPoint(this.players[playerName]["actor"]),
                    // Environment.getPoint(this.players[playerName]["actor"]),
                ]
            }
            // console.log(`${playerName} reward : ${Math.round(this.players[playerName]["point"] * 10000) / 10000}`)
        })
    }

    control(ctrlDatas, CP) {
        this.channel.postMessage({
            instruction: "ctrl",
            args: {
                archives: Object.keys(ctrlDatas).reduce(
                    (acc, playerName) => {
                        acc[playerName] = {
                            state: this.players[playerName]["memory"].slice(0, this.ctrlLength),
                            rewards: this.players[playerName]["point"].map((point) => point / this.steps),
                            actions: [
                                [
                                    this.players[playerName]["actor"].keyDown.jump,
                                    this.players[playerName]["actor"].keyDown.squat,
                                    (this.players[playerName]["actor"].keyDown.left && this.players[playerName]["actor"].shouldFaceTo == "left") ||
                                    (this.players[playerName]["actor"].keyDown.right && this.players[playerName]["actor"].shouldFaceTo == "right"),
                                    (this.players[playerName]["actor"].keyDown.left && this.players[playerName]["actor"].shouldFaceTo == "right") ||
                                    (this.players[playerName]["actor"].keyDown.right && this.players[playerName]["actor"].shouldFaceTo == "left"),
                                    this.players[playerName]["actor"].keyDown.attack.small,//|| this.players[playerName]["actor"]._state["subsection"] == "small",
                                    this.players[playerName]["actor"].keyDown.attack.medium,//|| this.players[playerName]["actor"]._state["subsection"] == "medium",
                                    this.players[playerName]["actor"].keyDown.attack.large,//|| this.players[playerName]["actor"]._state["subsection"] == "large" || this.players[playerName]["actor"]._state["subsection"] == "fall"
                                ].reduce((prev, curr, idx) => prev + curr * (2 ** idx), 0)
                            ],
                            discount: 0.65 + ((Math.abs(this.players[playerName]["actor"].mesh.position.x - this.players[playerName]["actor"].opponent.mesh.position.x) / 22) ** 0.5) * 0.34,
                            chooseActionRandomValue: ctrlDatas[playerName].chooseActionRandomValue,
                            aiCtrl: ctrlDatas[playerName].aiCtrl
                        }

                        return acc
                    }, {}),
                CP: CP
            }
        })
        this.steps = 0
        Object.keys(ctrlDatas).forEach(
            (playerName) => {
                this.players[playerName]["point"] = 0
            })
    }

    train(bsz = 32, replayIdxes = [null], usePrioritizedReplay = false) {
        this.channel.postMessage({
            instruction: "train",
            args: {
                bsz: bsz,
                replayIdxes: replayIdxes,
                usePrioritizedReplay: usePrioritizedReplay
            }
        })
        // console.log("train")
    }

    init() {
        Object.keys(this.players).forEach((playerName) => {
            this.trigger(playerName, 8)
        })
        Object.values(this.players).forEach((player) => {
            player.memory = new Array(this.memorySize).fill(Environment.getState(player))
            player.reward = 0
            player.action = "null"
        })

        console.log("init")
    }
    save() {
        this.channel.postMessage({
            instruction: "save",
            args: {}
        })
        // console.log("save")
    }
    load() {
        tf.tidy(() => {
            let load = document.createElement("input")
            load.type = "file"
            load.accept = ".bin"

            load.onchange = event => {
                const files = load.files
                // console.log(files[0])
                var reader = new FileReader()
                reader.addEventListener("loadend", () => {
                    this.channel.postMessage({
                        instruction: "load",
                        args: {
                            weightsBuffer: new Uint8Array(reader.result)
                        }
                    })
                    // console.log("load")
                });
                reader.readAsArrayBuffer(files[0])
            };

            load.click()
        })
    }
    updatePrioritys() {
        this.channel.postMessage({
            instruction: "updatePrioritys"
        })
    }

    static getState(player) {
        //actorA state
        let getActorState = (actor) => {
            let HP = actor.HP / actor.maxHP
            let cumulativeDamage = actor.cumulativeDamage / actor.maxCumulativeDamage
            let position = {
                x: actor.mesh.position.x / 11,
                y: actor.mesh.position.y / 11
            }
            let faceTo = actor._faceTo == actor.shouldFaceTo ? 1 : -1
            let jumpAttackNum = actor.jumpAttackNum / 5
            let jumpTimes = actor.jumpTimes / 2


            let chapter = new Array(4).fill(-1 * actor._state["frame"])
            switch (actor._state["chapter"]) {
                case "normal":
                    {
                        chapter[0] = actor._state["frame"]
                        break
                    }
                case "attack":
                    {
                        chapter[1] = actor._state["frame"]
                        break
                    }
                case "defense":
                    {
                        chapter[2] = actor._state["frame"]
                        break
                    }
                case "hitRecover":
                    {
                        chapter[3] = actor._state["frame"]
                        break
                    }
            }

            let section = new Array(4).fill(0 * actor._state["frame"])
            switch (actor._state["section"]) {
                case "stand":
                    {
                        section[0] = actor._state["frame"]
                        break
                    }
                case "jump":
                    {
                        section[1] = actor._state["frame"]
                        break
                    }
                case "squat":
                    {
                        section[2] = actor._state["frame"]
                        break
                    }
                case "reStand":
                    {
                        section[3] = actor._state["frame"]
                        break
                    }
            }
            if (actor.lastAttack != null) {
                switch (actor.lastAttack.split(":")[0]) {
                    case "stand":
                        {
                            section[0] = (0.1 + cumulativeDamage) * -1
                            break
                        }
                    case "jump":
                        {
                            section[1] = (0.1 + cumulativeDamage) * -1
                            break
                        }
                    case "squat":
                        {
                            section[2] = (0.1 + cumulativeDamage) * -1
                            break
                        }
                }
            }

            let subsection = new Array(7).fill(0 * actor._state["frame"])
            switch (actor._state["subsection"]) {
                case "main":
                    {
                        subsection[0] = actor._state["frame"]
                        break
                    }
                case "forward":
                    {
                        subsection[1] = actor._state["frame"]
                        break
                    }
                case "backward":
                    {
                        subsection[2] = actor._state["frame"]
                        break
                    }
                case "small":
                    {
                        subsection[3] = actor._state["frame"]
                        break
                    }
                case "medium":
                    {
                        subsection[4] = actor._state["frame"]
                        break
                    }
                case "large":
                    {
                        subsection[5] = actor._state["frame"]
                        break
                    }
                case "fall":
                    {
                        subsection[6] = actor._state["frame"]
                        break
                    }
            }
            if (actor.lastAttack != null) {
                switch (actor.lastAttack.split(":")[1]) {
                    case "small":
                        {
                            subsection[3] = (0.1 + cumulativeDamage) * -1
                            break
                        }
                    case "medium":
                        {
                            subsection[4] = (0.1 + cumulativeDamage) * -1
                            break
                        }
                    case "large":
                        {
                            subsection[5] = (0.1 + cumulativeDamage) * -1
                            break
                        }
                }
            }

            let subsubsection = new Array(4).fill(-1 * actor._state["frame"])
            subsubsection[actor._state["subsubsection"]] = actor._state["frame"]

            return [HP, cumulativeDamage, position.x, position.y, faceTo, jumpAttackNum, jumpTimes]
                .concat(chapter)
                .concat(section)
                .concat(subsection)
                .concat(subsubsection)
        }

        return getActorState(player["actor"])
            .concat([
                player["actor"].keyDown.jump,
                player["actor"].keyDown.squat,
                (player["actor"].keyDown.left && player["actor"].shouldFaceTo == "left") ||
                (player["actor"].keyDown.right && player["actor"].shouldFaceTo == "right"),
                (player["actor"].keyDown.left && player["actor"].shouldFaceTo == "right") ||
                (player["actor"].keyDown.right && player["actor"].shouldFaceTo == "left"),
                player["actor"].keyDown.attack.small,
                player["actor"].keyDown.attack.medium,
                player["actor"].keyDown.attack.large,
            ])
            .concat(getActorState(player["actor"].opponent))
        // .concat([
        //     player["actor"].opponent.keyDown.jump,
        //     player["actor"].opponent.keyDown.squat,
        //     (player["actor"].opponent.keyDown.left && player["actor"].opponent.shouldFaceTo == "left") ||
        //     (player["actor"].opponent.keyDown.right && player["actor"].opponent.shouldFaceTo == "right"),
        //     (player["actor"].opponent.keyDown.left && player["actor"].opponent.shouldFaceTo == "right") ||
        //     (player["actor"].opponent.keyDown.right && player["actor"].opponent.shouldFaceTo == "left"),
        //     player["actor"].opponent.keyDown.attack.small,
        //     player["actor"].opponent.keyDown.attack.medium,
        //     player["actor"].opponent.keyDown.attack.large,
        // ])
    }

    static getPoint(actor) {
        let point = ((actor.HP - actor.cumulativeDamage) / actor.maxHP) - ((actor.opponent.HP - actor.opponent.cumulativeDamage) / actor.opponent.maxHP) - 0.25

        if (actor._state["chapter"] == "hitRecover") {
            point -= (1 - ((actor.HP - actor.cumulativeDamage) / actor.maxHP)) * 10
        }
        if (actor.opponent._state["chapter"] == "hitRecover" && actor._state["chapter"] == "attack") {
            point += (1 - ((actor.opponent.HP - actor.opponent.cumulativeDamage) / actor.opponent.maxHP)) * 5
        }

        return point
    }

    static getMovePoint(actor) {
        let point = 0

        if (actor.keyDown.left == true && actor.keyDown.right == true) {
            point = -2 * Math.abs(Environment.getPoint(actor))
        }

        return point
    }
}