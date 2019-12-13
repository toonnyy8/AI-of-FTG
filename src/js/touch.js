import * as Hammer from "hammerjs"

export let touchAction = (player, keySet, canvas) => {
    let _keySet = Object.keys(keySet)
        .reduce((last, actionName) => {
            if (actionName == "attack") {
                last[actionName] = Object.keys(keySet["attack"])
                    .reduce((last, attackName) => {
                        last[attackName] = {
                            keyup: new KeyboardEvent("keyup", {
                                key: keySet["attack"][attackName]
                            }),
                            keydown: new KeyboardEvent("keydown", {
                                key: keySet["attack"][attackName]
                            })
                        }
                        return last
                    }, {})
            } else {
                last[actionName] = {
                    keyup: new KeyboardEvent("keyup", {
                        key: keySet[actionName]
                    }),
                    keydown: new KeyboardEvent("keydown", {
                        key: keySet[actionName]
                    })
                }
            }
            return last
        }, {})

    // Create a manager to manager the element
    let manager = new Hammer.Manager(canvas)

    // Create a recognizer
    let PanH = new Hammer.Pan({
        event: "panh",
        threshold: 10,
        direction: Hammer.DIRECTION_HORIZONTAL
    })
    let PanV = new Hammer.Pan({
        event: "panv",
        threshold: 20,
        direction: Hammer.DIRECTION_VERTICAL
    })
    let Swipe = new Hammer.Swipe({
        threshold: 5,
        velocity: 0
    })
    let singleTap = new Hammer.Tap({ event: 'singletap', time: 100 })
    let doubleTap = new Hammer.Tap({ event: 'doubletap', time: 250 })
    let tripleTap = new Hammer.Tap({ event: 'tripletap', time: 1000 })
    PanH.recognizeWith(PanV)
    PanV.recognizeWith(PanH)
    Swipe.recognizeWith(PanH)
    Swipe.recognizeWith(PanV)
    // Add the recognizer to the manager
    manager.add(PanH)
    manager.add(PanV)
    manager.add(Swipe)
    manager.add(singleTap)
    manager.add(doubleTap)
    manager.add(tripleTap)

    // Declare global letiables to swiped correct distance
    let jump = false
    let squat = false

    // Subscribe to a desired event
    manager.on("panh", function (e) {
        switch (e.offsetDirection) {
            case 2: {
                document.dispatchEvent(_keySet["right"].keyup)
                document.dispatchEvent(_keySet["left"].keydown)
                break
            }
            case 4: {
                document.dispatchEvent(_keySet["left"].keyup)
                document.dispatchEvent(_keySet["right"].keydown)
                break
            }
        }
    })
    manager.on("panv", function (e) {
        switch (e.offsetDirection) {
            case 8: {
                if (!jump) {
                    document.dispatchEvent(_keySet["squat"].keyup)
                    squat = false
                    document.dispatchEvent(_keySet["jump"].keydown)
                    jump = true
                }
                break
            }
            case 16: {
                if (player.mesh.position.y > 0) {
                    document.dispatchEvent(_keySet["squat"].keydown)
                    squat = true
                    document.dispatchEvent(_keySet["attack"]["large"].keydown)
                    setTimeout(() => {
                        document.dispatchEvent(_keySet["attack"]["large"].keyup)
                    }, 250)
                }
            }
        }
    })

    manager.on("swipe", function (e) {
        console.log("swipe")
        switch (e.offsetDirection) {
            case 16: {
                if (!squat) {
                    document.dispatchEvent(_keySet["squat"].keydown)
                    squat = true
                } else {
                    document.dispatchEvent(_keySet["squat"].keyup)
                    squat = false
                }
                break
            }
        }
        document.dispatchEvent(_keySet["left"].keyup)
        document.dispatchEvent(_keySet["right"].keyup)
        document.dispatchEvent(_keySet["jump"].keyup)
        jump = false
    })

    manager.on("singletap", function (e) {
        document.dispatchEvent(_keySet["attack"]["small"].keydown)
        setTimeout(() => {
            document.dispatchEvent(_keySet["attack"]["small"].keyup)
        }, 250)
    })

    manager.on("doubletap", function (e) {
        document.dispatchEvent(_keySet["attack"]["medium"].keydown)
        setTimeout(() => {
            document.dispatchEvent(_keySet["attack"]["medium"].keyup)
        }, 250)
    })

    manager.on("tripletap", function (e) {
        document.dispatchEvent(_keySet["attack"]["large"].keydown)
        setTimeout(() => {
            document.dispatchEvent(_keySet["attack"]["large"].keyup)
        }, 250)
    })

}
