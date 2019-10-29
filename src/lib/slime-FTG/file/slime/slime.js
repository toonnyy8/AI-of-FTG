import fs from 'fs'

import * as BABYLON from "babylonjs"

export class Actor {
    constructor({
        mesh,
        materialMesh,
        animationGroup,
        skeleton,
        startPosition,
        startRotationQuaternion,
        scene,
        keySet = {
            jump: "w",
            squat: "s",
            left: "a",
            right: "d",
            attack: {
                small: "j",
                medium: "k",
                large: "l"
            }
        },
        fps = 60,
        maxHP = 3000,
        maxCumulativeDamage = 500
    }) {
        this._faceTo = "left"
        this._fps = fps && !Number.isNaN(fps - 0) ? fps : this.fps
        this._actions = Actor.actionSet()
        this._state = { chapter: "normal", section: "stand", subsection: "main", subsubsection: 0, frame: 0 }
        this._animationGroup = animationGroup
        this._mesh = mesh
        this._materialMesh = materialMesh
        this._skeleton = skeleton
        this._startPosition = startPosition
        this._startRotationQuaternion = startRotationQuaternion
        this._scene = scene
        this._opponent = null
        this.keyDown = {
            jump: false,
            squat: false,
            left: false,
            right: false,
            attack: {
                small: false,
                medium: false,
                large: false
            }
        }
        this.jumpAttackNum = 0
        this.isHit = false
        this.jumpTimes = 0

        this.maxHP = maxHP
        this.HP = this.maxHP
        this.beHitNum = 0

        this.vector = BABYLON.Vector3.Zero()
        this.mesh.position = this._startPosition.clone()
        this.mesh.rotationQuaternion = this._startRotationQuaternion.clone()

        this.cumulativeDamage = 0
        this.maxCumulativeDamage = 500

        this.perfectDefenseTime = -1
        this.isPD = false

        this.beInjuredObj = { atk: null, scale: null, beHitVector: BABYLON.Vector3.Zero() }

        this.material = this.materialMesh.material

        //collision boxes
        this._collisionBoxes = []
        this.skeleton.bones.forEach((bone, index) => {
            let box = new BABYLON.MeshBuilder.CreateBox("box", { size: 0.1, updatable: true }, this.scene)
            box.PhysicsImpostor = new BABYLON.PhysicsImpostor(box, BABYLON.PhysicsImpostor.SphereImpostor, { mass: 0 }, this.scene)
            box.material = new BABYLON.StandardMaterial("myMaterial", this.scene);
            box.material.alpha = 0
            this._collisionBoxes.push(box)
        })
        this._bodyBox = new BABYLON.MeshBuilder.CreateSphere("sphere", { diameter: 1.8, updatable: true }, this.scene)
        this._bodyBox.setPivotMatrix(new BABYLON.Matrix.Translation(0, 0.5, 0), false);
        this._bodyBox.position = this.mesh.position
        this._bodyBox.material = new BABYLON.StandardMaterial("myMaterial", this.scene);
        this._bodyBox.material.alpha = 0

        //animatiom
        Object.keys(this._actions).forEach(chapter => {
            Object.keys(this._actions[chapter]).forEach(section => {
                Object.keys(this._actions[chapter][section]).forEach(subsection => {
                    this._actions[chapter][section][subsection].forEach((anim, subsubsection, animsArray) => {
                        animsArray[subsubsection] = animationGroup.clone()
                        // console.log(`${chapter}:${section}:${subsection}:${subsubsection}`)
                        animsArray[subsubsection].normalize(Actor.actionSet()[chapter][section][subsection][subsubsection].start / this.fps, Actor.actionSet()[chapter][section][subsection][subsubsection].end / this.fps)
                    })
                    let stateEqual = (subsubsection) => {
                        return `${chapter}:${section}:${subsection}:${subsubsection}` == `${this._state["chapter"]}:${this._state["section"]}:${this._state["subsection"]}:${this._state["subsubsection"]}`
                    }
                    switch (chapter) {
                        case "normal":
                            {
                                switch (section) {
                                    case "squat":
                                        {
                                            switch (subsection) {
                                                case "main":
                                                    {
                                                        this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                            if (this._state.chapter == "normal") {
                                                                if (this._state.section == "squat") {
                                                                    if (this.keyDown.squat) {
                                                                        this._state.subsubsection = 1
                                                                    } else {
                                                                        this._state.subsubsection = 2
                                                                    }
                                                                }
                                                            }
                                                        })
                                                        this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                            this._state.subsubsection = 0
                                                            this._state.section = "stand"
                                                        })
                                                        break;
                                                    }
                                                default:
                                                    break;
                                            }
                                            break;
                                            break;
                                        }
                                    case "jump":
                                        {
                                            switch (subsection) {
                                                case "main":
                                                    {
                                                        // this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                        //     if (this._state.section == "jump") {
                                                        //         this._state.subsubsection = 1
                                                        //     }
                                                        // })
                                                        // this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                        //     this._state.subsubsection = 0
                                                        //     this._state.section = "stand"
                                                        // })
                                                        break;
                                                    }
                                                default:
                                                    break;
                                            }
                                            break;
                                            break;
                                        }
                                    default:
                                        break;
                                }
                                break;
                            }
                        case "attack":
                            {
                                this._actions[chapter][section][subsection][0].onAnimationGroupPlayObservable.add(() => {
                                    setTimeout(() => { this.isHit = false }, 30)
                                })

                                switch (section) {
                                    case "stand":
                                        {
                                            if (subsection != "large") {
                                                this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(0)) {
                                                        if (this.isHit) {
                                                            this._state.subsubsection = 1
                                                        } else {
                                                            this._state.subsubsection = 2
                                                        }
                                                    }
                                                })
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        this._state.subsubsection = 2
                                                    }

                                                })
                                                this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(2)) {
                                                        this._state.subsubsection = 0
                                                        this._state.chapter = "normal"
                                                        this._state.subsection = "main"
                                                        this.isHit = false
                                                    }
                                                })
                                            } else {
                                                this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(0)) {
                                                        this._state.subsubsection = 1
                                                    }
                                                })
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        if (this.isHit) {
                                                            this._state.subsubsection = 2
                                                        } else {
                                                            this._state.subsubsection = 3
                                                        }
                                                    }
                                                })
                                                this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(2)) {
                                                        this._state.subsubsection = 3
                                                    }

                                                })
                                                this._actions[chapter][section][subsection][3].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(3)) {
                                                        this._state.subsubsection = 0
                                                        this._state.chapter = "normal"
                                                        this._state.subsection = "main"
                                                        this.isHit = false
                                                    }
                                                })
                                            }
                                            break;
                                        }
                                    case "squat":
                                        {
                                            if (subsection == "small") {
                                                this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(0)) {
                                                        this._state.subsubsection = 1
                                                        this.vector.x = this.shouldFaceTo == "left" ? 1 : -1
                                                    }
                                                })
                                            } else if (subsection == "large") {
                                                this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(0)) {
                                                        this._state.subsubsection = 1
                                                    }
                                                })
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        if (this.isHit) {
                                                            this._state.subsubsection = 2
                                                        } else {
                                                            this._state.subsubsection = 3
                                                        }
                                                    }
                                                })
                                            } else {
                                                this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(0)) {
                                                        if (this.isHit) {
                                                            this._state.subsubsection = 1
                                                        } else {
                                                            this._state.subsubsection = 2
                                                        }
                                                    }
                                                })
                                            }
                                            if (subsection == "large") {
                                                this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(2)) {
                                                        this._state.subsubsection = 3
                                                    }
                                                })
                                                this._actions[chapter][section][subsection][3].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(3)) {
                                                        this._state.chapter = "normal"
                                                        this._state.subsection = "main"
                                                        if (this.keyDown.squat) {
                                                            this._state.subsubsection = 1
                                                        } else {
                                                            this._state.subsubsection = 2
                                                        }
                                                        this.isHit = false
                                                    }
                                                })
                                            } else {
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        this._state.subsubsection = 2
                                                    }
                                                })
                                                this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(2)) {
                                                        this._state.chapter = "normal"
                                                        this._state.subsection = "main"
                                                        if (this.keyDown.squat) {
                                                            this._state.subsubsection = 1
                                                        } else {
                                                            this._state.subsubsection = 2
                                                        }
                                                        this.isHit = false
                                                    }
                                                })
                                            }



                                            break;
                                        }
                                    case "jump":
                                        {
                                            switch (subsection) {
                                                case "small":
                                                    {
                                                        this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(0)) {
                                                                if (this.isHit) {
                                                                    this._state.subsubsection = 1
                                                                } else {
                                                                    this._state.subsubsection = 2
                                                                }
                                                            }
                                                        })

                                                        this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(1)) {
                                                                this._state.subsubsection = 2
                                                            }
                                                        })
                                                        this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(2)) {
                                                                this._state.chapter = "normal"
                                                                this._state.subsection = "main"
                                                                this._state.subsubsection = 0
                                                                this.isHit = false
                                                            }
                                                        })
                                                        break;
                                                    }
                                                case "fall":
                                                    {
                                                        this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(0)) {
                                                                this._state.subsubsection = 1
                                                            }
                                                        })

                                                        this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(1)) {
                                                                if (this.mesh.position.y <= 0) {
                                                                    this._state.subsubsection = 2
                                                                } else {
                                                                    this.isHit = false
                                                                    this._state.subsubsection = 1
                                                                }
                                                            }
                                                        })
                                                        this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(2)) {
                                                                this._state.chapter = "normal"
                                                                this._state.subsection = "main"
                                                                this._state.subsubsection = 0
                                                                this.isHit = false
                                                            }
                                                        })
                                                        break;
                                                    }
                                                default:
                                                    {
                                                        this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(0)) {
                                                                this._state.subsubsection = 1
                                                            }
                                                        })
                                                        this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(1)) {
                                                                this._state.subsubsection = 2
                                                            }
                                                        })

                                                        this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                            if (stateEqual(2)) {
                                                                this._state.chapter = "normal"
                                                                this._state.subsection = "main"
                                                                this._state.subsubsection = 0
                                                                this.isHit = false
                                                            }
                                                        })
                                                        break;
                                                    }
                                            }


                                            break;
                                        }
                                    default:
                                        break;
                                }
                                break;
                            }
                        case "hitRecover":
                            {
                                switch (section) {
                                    case "stand":
                                        {
                                            this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                if (stateEqual(0)) {
                                                    if (this.cumulativeDamage < this.maxCumulativeDamage) {
                                                        this._state.subsubsection = 1
                                                    } else {
                                                        this._state.subsection = "large"
                                                        this._state.subsubsection = 1
                                                    }
                                                }
                                            })

                                            if (subsection != "large") {
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        this._state.subsubsection = 0
                                                        this._state.chapter = "normal"
                                                        this._state.subsection = "main"
                                                        this.beHitNum = 0
                                                    }
                                                })
                                            } else {
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        this._state.section = "reStand"
                                                        this._state.subsection = "main"
                                                        this._state.subsubsection = 0
                                                        this.beHitNum = 0
                                                    }
                                                })
                                            }

                                            break;
                                        }
                                    case "squat":
                                        {
                                            this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                if (stateEqual(0)) {
                                                    if (this.cumulativeDamage < this.maxCumulativeDamage) {
                                                        this._state.subsubsection = 1
                                                    } else {
                                                        this._state.subsection = "large"
                                                        this._state.subsubsection = 1
                                                    }
                                                }

                                            })

                                            if (subsection != "large") {
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        this._state.subsubsection = 1
                                                        this._state.chapter = "normal"
                                                        this._state.subsection = "main"
                                                    }
                                                })
                                            } else {
                                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                    if (stateEqual(1)) {
                                                        this._state.section = "reStand"
                                                        this._state.subsection = "main"
                                                        this._state.subsubsection = 0
                                                    }
                                                })
                                            }

                                            break;
                                        }
                                    case "jump":
                                        {
                                            this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                if (stateEqual(0)) {
                                                    this._state.subsubsection = 1
                                                }
                                            })
                                            this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                if (stateEqual(1)) {
                                                    this._state.subsubsection = 2
                                                }
                                            })
                                            this._actions[chapter][section][subsection][2].onAnimationEndObservable.add(() => {
                                                if (stateEqual(2)) {
                                                    this._state.subsubsection = 3
                                                }
                                            })

                                            this._actions[chapter][section][subsection][3].onAnimationEndObservable.add(() => {
                                                if (stateEqual(3)) {
                                                    this._state.section = "reStand"
                                                    this._state.subsection = "main"
                                                    this._state.subsubsection = 0
                                                    this.beHitNum = 0
                                                }
                                            })

                                            break;
                                        }
                                    case "reStand":
                                        {
                                            this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                                if (stateEqual(0)) {
                                                    this._state.subsubsection = 1
                                                }
                                            })
                                            this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                                if (stateEqual(1)) {
                                                    this._state.chapter = "normal"
                                                    this._state.subsection = "main"
                                                    if (this.keyDown.squat) {
                                                        this._state.section = "squat"
                                                    } else {
                                                        this._state.section = "stand"
                                                    }
                                                    this._state.subsubsection = 0
                                                    this.beHitNum = 0
                                                }
                                            })
                                            break;
                                        }
                                    default:
                                        break;
                                }
                                break;
                            }
                        case "defense":
                            {
                                this._actions[chapter][section][subsection][0].onAnimationGroupPlayObservable.add(() => {
                                    setTimeout(() => {
                                        this.materialMesh.material = this.material
                                    }, 100)
                                })

                                this._actions[chapter][section][subsection][0].onAnimationEndObservable.add(() => {
                                    if (stateEqual(0)) {
                                        this._state.subsubsection = 1
                                    }
                                })
                                this._actions[chapter][section][subsection][1].onAnimationEndObservable.add(() => {
                                    if (stateEqual(1)) {
                                        this._state.chapter = "normal"
                                        this._state.subsection = "main"
                                        if (this._state.section == "stand") {
                                            this._state.subsubsection = 0
                                        } else if (this._state.section == "squat") {
                                            if (this.keyDown.squat) {
                                                this._state.subsubsection = 1
                                            } else {
                                                this._state.subsubsection = 2
                                            }
                                        } else if (this._state.section == "jump") {
                                            this._state.subsubsection = 0
                                        }
                                        this.isPD = false

                                    }
                                })
                                break;
                            }
                        default:
                            break;
                    }

                })
            })
        })

        //key down
        document.addEventListener('keydown', (event) => {
            // console.log(event.key)
            switch (event.key) {
                case keySet.right:
                    {
                        if (!this.keyDown.right) {
                            if (this._state.chapter == "normal") {
                                if (this._state.section == "stand") {
                                    this._state.subsection = this.shouldFaceTo == "right" ? "forward" : "backward"
                                }
                                this.keyDown.right = true
                                if (this.shouldFaceTo == "left") {
                                    this.perfectDefenseTime = 6
                                }
                            }
                        }
                        break;
                    }
                case keySet.left:
                    {
                        if (!this.keyDown.left) {
                            if (this._state.chapter == "normal") {
                                if (this._state.section == "stand") {
                                    this._state.subsection = this.shouldFaceTo == "left" ? "forward" : "backward"
                                }
                                this.keyDown.left = true
                                if (this.shouldFaceTo == "right") {
                                    this.perfectDefenseTime = 6
                                }
                            }
                        }
                        break;
                    }
                case keySet.jump:
                    {
                        if (!this.keyDown.jump && this.jumpTimes < 2) {
                            if (this.cumulativeDamage < this.maxCumulativeDamage) {
                                if (this._state.chapter == "hitRecover" && this._state.section != "reStand" && this._state.subsubsection == 1) {
                                    this._state.chapter = "normal"
                                    this.HP -= this.cumulativeDamage * (this.HP / this.maxHP)
                                    this.cumulativeDamage = 0
                                }
                            }
                            if (this._state.chapter == "normal") {

                                this._state.section = "jump"
                                this._state.subsection = "main"
                                this._state.subsubsection = 0
                                this.keyDown.jump = true
                                this.vector.y = 0.4
                                this.mesh.position.y += 0.01
                                this.jumpTimes += 1

                                if (this.shouldFaceTo == "left") {
                                    this._faceTo = "left"
                                    this.mesh.rotationQuaternion = new BABYLON.Vector3(0, 0, 0).toQuaternion()
                                } else {
                                    this._faceTo = "right"
                                    this.mesh.rotationQuaternion = new BABYLON.Vector3(0, Math.PI, 0).toQuaternion()
                                }
                                if (this.keyDown.left != this.keyDown.right) {
                                    if (this.keyDown.left) {
                                        this.vector.x = this.shouldFaceTo == "left" ? 0.1 : 0.075
                                    } else if (this.keyDown.right) {
                                        this.vector.x = this.shouldFaceTo == "right" ? -0.1 : -0.075
                                    }
                                }
                            }
                        }
                        break;
                    }
                case keySet.squat:
                    {
                        if (!this.keyDown.squat) {
                            if (this._state.chapter == "normal") {
                                if (this._state.section == "stand") {
                                    this._state.section = "squat"
                                    this._state.subsection = "main"
                                    this._state.subsubsection = 0
                                }

                            }
                            this.keyDown.squat = true
                        }
                        break;
                    }
                case keySet.attack.small:
                    {
                        if (!this.keyDown.attack.small) {
                            switch (this._state.chapter) {
                                case "normal":
                                    {
                                        if (this.jumpAttackNum < 5) {
                                            this._state.chapter = "attack"
                                            this._state.subsection = "small"
                                            this._state.subsubsection = 0
                                            this.keyDown.attack.small = true
                                            if (this._state.section == "jump") {
                                                this.jumpAttackNum += 1
                                            }
                                        }
                                        break;
                                    }
                                case "attack":
                                    {
                                        if (this._state.subsection != "small" && this._state.subsection != "fall") {
                                            if (this.isHit) {
                                                if (this._state.subsubsection == Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection].length - 1) {
                                                    this._state.chapter = "attack"
                                                    this._state.subsection = "small"
                                                    this._state.subsubsection = 0
                                                    this.keyDown.attack.small = true
                                                    if (this._state.section == "jump") {
                                                        this.jumpAttackNum += 1
                                                    }
                                                    // this.isHit = false
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case "hitRecover":
                                    {
                                        // if (this._state.section != "reStand") {
                                        //     if (this._state.subsubsection == Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection].length - 1) {
                                        //         if (this.jumpAttackNum < 5) {
                                        //             this._state.chapter = "attack"
                                        //             this._state.subsection = "small"
                                        //             this._state.subsubsection = 0
                                        //             this.keyDown.attack.small = true
                                        //             if (this._state.section == "jump") {
                                        //                 this.jumpAttackNum += 1
                                        //             }
                                        //         }
                                        //     }
                                        // }
                                        break;
                                    }
                                case "defense":
                                    {
                                        if (this.isPD) {
                                            if (this.jumpAttackNum < 5) {
                                                this._state.chapter = "attack"
                                                this._state.subsection = "small"
                                                this._state.subsubsection = 0
                                                this.keyDown.attack.small = true
                                                if (this._state.section == "jump") {
                                                    this.jumpAttackNum += 1
                                                }
                                            }
                                        }
                                        this.isPD = false
                                        break;
                                    }
                                default:
                                    break;
                            }
                        }
                        break;
                    }
                case keySet.attack.medium:
                    {
                        if (!this.keyDown.attack.medium) {
                            switch (this._state.chapter) {
                                case "normal":
                                    {
                                        if (this.jumpAttackNum < 5) {
                                            this._state.chapter = "attack"
                                            this._state.subsection = "medium"
                                            this._state.subsubsection = 0
                                            this.keyDown.attack.medium = true
                                            if (this._state.section == "jump") {
                                                this.jumpAttackNum += 2
                                            }
                                        }
                                        break;
                                    }
                                case "attack":
                                    {
                                        if (this._state.subsection != "medium" && this._state.subsection != "fall") {
                                            if (this.isHit) {
                                                if (this._state.subsubsection == Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection].length - 1) {
                                                    this._state.chapter = "attack"
                                                    this._state.subsection = "medium"
                                                    this._state.subsubsection = 0
                                                    this.keyDown.attack.medium = true
                                                    if (this._state.section == "jump") {
                                                        this.jumpAttackNum += 2
                                                    }
                                                    // this.isHit = false
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case "hitRecover":
                                    {
                                        break;
                                    }
                                case "defense":
                                    {
                                        if (this.isPD) {
                                            if (this.jumpAttackNum < 5) {
                                                this._state.chapter = "attack"
                                                this._state.subsection = "medium"
                                                this._state.subsubsection = 0
                                                this.keyDown.attack.medium = true
                                                if (this._state.section == "jump") {
                                                    this.jumpAttackNum += 2
                                                }
                                            }
                                        }
                                        this.isPD = false
                                        break;
                                    }
                                default:
                                    break;
                            }
                        }
                        break;
                    }
                case keySet.attack.large:
                    {
                        if (!this.keyDown.attack.large) {
                            switch (this._state.chapter) {
                                case "normal":
                                    {
                                        if (this.jumpAttackNum < 5) {
                                            this._state.chapter = "attack"
                                            if (this.keyDown.squat && this._state.section == "jump") {
                                                this._state.subsection = "fall"
                                            } else {
                                                this._state.subsection = "large"
                                            }
                                            this._state.subsubsection = 0
                                            this.keyDown.attack.large = true
                                            if (this._state.section == "jump") {
                                                this.jumpAttackNum += 3
                                            }
                                        }
                                        break;
                                    }
                                case "attack":
                                    {
                                        if (this._state.subsection != "large" && this._state.subsection != "fall") {
                                            if (this.isHit) {
                                                if (this._state.subsubsection == Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection].length - 1) {
                                                    this._state.chapter = "attack"
                                                    if (this.keyDown.squat && this._state.section == "jump") {
                                                        this._state.subsection = "fall"
                                                    } else {
                                                        this._state.subsection = "large"
                                                    }
                                                    this._state.subsubsection = 0
                                                    this.keyDown.attack.large = true
                                                    if (this._state.section == "jump") {
                                                        this.jumpAttackNum += 3
                                                    }
                                                    // this.isHit = false
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case "hitRecover":
                                    {
                                        break;
                                    }
                                case "defense":
                                    {
                                        if (this.isPD) {
                                            if (this.jumpAttackNum < 5) {
                                                this._state.chapter = "attack"
                                                if (this.keyDown.squat && this._state.section == "jump") {
                                                    this._state.subsection = "fall"
                                                } else {
                                                    this._state.subsection = "large"
                                                }
                                                this._state.subsubsection = 0
                                                this.keyDown.attack.large = true
                                                if (this._state.section == "jump") {
                                                    this.jumpAttackNum += 3
                                                }
                                            }
                                        }
                                        this.isPD = false
                                        break;
                                    }
                                default:
                                    break;
                            }
                        }
                        break;
                    }
                default:
                    break;
            }
        }, false)

        //key up
        document.addEventListener('keyup', (event) => {
            switch (event.key) {
                case keySet.right:
                    {
                        if (this.keyDown.right) {
                            if (this._state.chapter == "normal") {
                                this._state.subsection = "main"
                            }
                        }
                        if (this.keyDown.left) {
                            if (this._state.chapter == "normal") {
                                if (this._state.section == "stand") {
                                    this._state.subsection = this.shouldFaceTo == "left" ? "forward" : "backward"
                                }
                            }
                        }
                        this.keyDown.right = false
                        break;
                    }
                case keySet.left:
                    {
                        if (this.keyDown.left) {
                            if (this._state.chapter == "normal") {
                                this._state.subsection = "main"
                            }
                        }
                        if (this.keyDown.right) {
                            if (this._state.chapter == "normal") {
                                if (this._state.section == "stand") {
                                    this._state.subsection = this.shouldFaceTo == "right" ? "forward" : "backward"
                                }
                            }
                        }
                        this.keyDown.left = false
                        break;
                    }
                case keySet.jump:
                    {
                        this.keyDown.jump = false
                        break;
                    }
                case keySet.squat:
                    {
                        if (this.keyDown.squat) {
                            if (this._state.chapter == "normal") {
                                if (this._state.section == "squat") {
                                    if (this._state.subsubsection == 1) {
                                        this._state.subsubsection = 2
                                    }
                                }
                            }
                        }
                        this.keyDown.squat = false
                        break;
                    }
                case keySet.attack.small:
                    {
                        if (this.keyDown.attack.small) {

                        }
                        this.keyDown.attack.small = false
                        break;
                    }
                case keySet.attack.medium:
                    {
                        this.keyDown.attack.medium = false
                        break;
                    }
                case keySet.attack.large:
                    {
                        this.keyDown.attack.large = false
                        break;
                    }
                default:
                    break;
            }
        }, false)


    }
    static actionSet() {
        return {
            normal: {
                stand: {
                    main: [{
                        start: 1,
                        end: 40,
                        atk: 0
                    }],
                    forward: [{
                        start: 41,
                        end: 70,
                        atk: 0
                    }],
                    backward: [{
                        start: 71,
                        end: 100,
                        atk: 0
                    }]
                },
                jump: {
                    main: [
                        //     {
                        //     start: 401,
                        //     end: 410,
                        //     atk: 0
                        // },
                        {
                            start: 411,
                            end: 430,
                            atk: 0
                        },
                        // {
                        //     start: 431,
                        //     end: 440,
                        //     atk: 0
                        // }
                    ]
                },
                squat: {
                    main: [{
                        start: 801,
                        end: 820,
                        atk: 0,
                        speed: 3
                    },
                    {
                        start: 821,
                        end: 880,
                        atk: 0
                    },
                    {
                        start: 1101,
                        end: 1120,
                        atk: 0,
                        speed: 3
                    }
                    ],
                    // forward: [{
                    //     start: 881,
                    //     end: 920,
                    //     atk: 0
                    // }],
                    // backward: [{
                    //     start: 921,
                    //     end: 960,
                    //     atk: 0
                    // }]
                }
            },
            attack: {
                stand: {
                    small: [{
                        start: 101,
                        end: 109,
                        atk: 100,
                        speed: 1.5,
                        boxes: [12]
                    }, {
                        start: 109,
                        end: 119,
                        atk: 0,
                        speed: 2
                    }, {
                        start: 119,
                        end: 130,
                        atk: 0,
                        speed: 1.5
                    }],
                    medium: [{
                        start: 131,
                        end: 150,
                        atk: 200,
                        speed: 2,
                        boxes: [7]
                    }, {
                        start: 150,
                        end: 160,
                        atk: 0,
                        speed: 2
                    }, {
                        start: 160,
                        end: 170,
                        atk: 0
                    }],
                    large: [{
                        start: 171,
                        end: 185,
                        atk: 0
                    }, {
                        start: 185,
                        end: 197,
                        atk: 300,
                        boxes: [12]
                    }, {
                        start: 197,
                        end: 207,
                        atk: 0,
                        speed: 2
                    }, {
                        start: 207,
                        end: 220,
                        atk: 0
                    }]
                },
                jump: {
                    small: [{
                        start: 441,
                        end: 450,
                        atk: 150,
                        speed: 1.5,
                        boxes: [12]
                    }, {
                        start: 450,
                        end: 460,
                        atk: 0,
                        speed: 2
                    }, {
                        start: 460,
                        end: 470,
                        atk: 0,
                        speed: 1.5
                    }],
                    medium: [{
                        start: 471,
                        end: 485,
                        atk: 0,
                        speed: 1.5
                    }, {
                        start: 485,
                        end: 493,
                        atk: 240,
                        speed: 1.5,
                        boxes: [12],
                        hitVector: [0, 0.25, 0]
                    }, {
                        start: 493,
                        end: 510,
                        atk: 0,
                    }],
                    large: [{
                        start: 511,
                        end: 520,
                        atk: 0,
                        speed: 1.8
                    }, {
                        start: 520,
                        end: 555,
                        atk: 500,
                        speed: 4,
                        boxes: [7, 6],
                        hitVector: [0, -0.5, 0]
                    }, {
                        start: 555,
                        end: 570,
                        atk: 0,
                        speed: 1.2
                    }],
                    fall: [{
                        start: 571,
                        end: 580,
                        atk: 75,
                        boxes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        hitVector: [-0.1, 0, 0]
                    }, {
                        start: 581,
                        end: 600,
                        atk: 75,
                        speed: 10,
                        boxes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        hitVector: [-0.1, 0, 0]
                    }, {
                        start: 601,
                        end: 630,
                        atk: 20,
                        speed: 2,
                        boxes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        hitVector: [-0.1, 0, 0]
                    }]
                },
                squat: {
                    small: [{
                        start: 961,
                        end: 970,
                        atk: 0
                    }, {
                        start: 970,
                        end: 985,
                        atk: 100,
                        speed: 2,
                        boxes: ["body"],
                        hitVector: [0, 0.2, 0]
                    }, {
                        start: 985,
                        end: 1000,
                        atk: 0
                    }],
                    medium: [{
                        start: 1001,
                        end: 1018,
                        atk: 250,
                        boxes: [12],
                        speed: 1.5,
                        hitVector: [0.05, 0.2, 0]
                    }, {
                        start: 1018,
                        end: 1030,
                        atk: 0,
                        speed: 2
                    }, {
                        start: 1030,
                        end: 1040,
                        atk: 0
                    }],
                    large: [{
                        start: 1041,
                        end: 1051,
                        atk: 0
                    },
                    {
                        start: 1051,
                        end: 1058,
                        atk: 350,
                        boxes: [12],
                        hitVector: [0.00001, 0.32, 0]
                    },
                    {
                        start: 1058,
                        end: 1071,
                        atk: 0,
                        speed: 2
                    }, {
                        start: 1071,
                        end: 1100,
                        atk: 0
                    }
                    ]
                }
            },
            defense: {
                stand: {
                    main: [{
                        start: 221,
                        end: 230,
                        atk: 0
                    }, {
                        start: 231,
                        end: 240,
                        atk: 0
                    }]
                },
                jump: {
                    main: [{
                        start: 631,
                        end: 640,
                        atk: 0
                    }, {
                        start: 640,
                        end: 650,
                        atk: 0
                    }]
                },
                squat: {
                    main: [{
                        start: 1121,
                        end: 1130,
                        atk: 0
                    }, {
                        start: 1130,
                        end: 1140,
                        atk: 0
                    }]
                }
            },
            hitRecover: {
                stand: {
                    small: [{
                        start: 241,
                        end: 250,
                        atk: 0,
                        speed: 0.8
                    }, {
                        start: 250,
                        end: 260,
                        atk: 0
                    }],
                    medium: [{
                        start: 261,
                        end: 275,
                        atk: 0,
                        speed: 0.8
                    }, {
                        start: 275,
                        end: 290,
                        atk: 0
                    }],
                    large: [{
                        start: 291,
                        end: 310,
                        atk: 0,
                        speed: 0.8
                    }, {
                        start: 310,
                        end: 330,
                        atk: 0
                    }]
                },
                jump: {
                    large: [{
                        start: 651,
                        end: 652,
                        atk: 0,
                        speed: 0.05
                    }, {
                        start: 652,
                        end: 680,
                        atk: 0,
                        speed: 1
                    }, {
                        start: 680,
                        end: 681,
                        atk: 0,
                        speed: 0.01
                    }, {
                        start: 681,
                        end: 700,
                        atk: 0
                    }]
                },
                squat: {
                    small: [{
                        start: 1141,
                        end: 1150,
                        atk: 0,
                        speed: 0.5
                    }, {
                        start: 1150,
                        end: 1160,
                        atk: 0
                    }],
                    medium: [{
                        start: 1161,
                        end: 1175,
                        atk: 0,
                        speed: 0.5
                    }, {
                        start: 1175,
                        end: 1190,
                        atk: 0
                    }],
                    large: [{
                        start: 1191,
                        end: 1210,
                        atk: 0,
                        speed: 0.5
                    }, {
                        start: 1210,
                        end: 1230,
                        atk: 0
                    }]
                },
                reStand: {
                    main: [{
                        start: 1300,
                        end: 1301,
                        atk: 0,
                        speed: 0.1
                    },
                    {
                        start: 1301,
                        end: 1330,
                        atk: 0
                    }
                    ]
                }
            }
        }
    }
    static url() {
        return URL.createObjectURL(new Blob([fs.readFileSync(__dirname + '../../../file/slime/slime.glb')]))
    }
    get fps() {
        return this._fps || 60
    }
    get animationGroup() {
        return this._animationGroup
    }
    get mesh() {
        return this._mesh
    }
    get materialMesh() {
        return this._materialMesh
    }
    get skeleton() {
        return this._skeleton
    }
    get scene() {
        return this._scene
    }
    get collisionBoxes() {
        return this._collisionBoxes
    }
    get bodyBox() {
        return this._bodyBox
    }

    stopAnimation() {
        Object.keys(this._actions).forEach((chapter => {
            Object.keys(this._actions[chapter]).forEach((section => {
                Object.keys(this._actions[chapter][section]).forEach((subsection => {
                    this._actions[chapter][section][subsection].forEach((anim, subsubsection) => {
                        if (`${chapter}:${section}:${subsection}:${subsubsection}` != `${this._state["chapter"]}:${this._state["section"]}:${this._state["subsection"]}:${this._state["subsubsection"]}`) {
                            anim.stop()
                        } else { }
                    })
                }))
            }))
        }))
    }

    tick(debug) {
        this.perfectDefenseTime -= 1

        if (this.cumulativeDamage < this.maxCumulativeDamage) {
            this.cumulativeDamage = this.cumulativeDamage <= 0 ? 0 : this.cumulativeDamage - 1
        }
        if (debug) {
            // console.log(this.vector)
        }
        if (debug) {
            console.log(`${this._state.chapter}:${this._state.section}:${this._state.subsection}:${this._state.subsubsection}`)
        }

        this.stopAnimation()
        try {
            this._actions[this._state.chapter][this._state.section][this._state.subsection][this._state.subsubsection].start(false, (Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection][this._state.subsubsection].speed || 1) /* * 0.5*/)
            this._state.frame = this._actions[this._state.chapter][this._state.section][this._state.subsection][this._state.subsubsection].animatables[0].getAnimations()[0].currentFrame

        } catch{
            console.error(`${this._state.chapter}:${this._state.section}:${this._state.subsection}:${this._state.subsubsection}`)
        }
        // if (`${this._state["chapter"]}:${this._state["section"]}:${this._state["subsection"]}` == "normal:stand:main") {
        //     this.vector.x = 0
        // }
        if (this.mesh.position.y == 0 && this._state.chapter != "attack") {
            this.vector.x = 0
        }

        if (this.mesh.position.y > 0) {
            this._state.section = "jump"
        }


        switch (this._state.chapter) {
            case "normal":
                {
                    this.isHit = false
                    // this.cumulativeDamage = 0
                    switch (this._state.section) {
                        case "stand":
                            {
                                this.jumpAttackNum = 0
                                this.jumpTimes = 0
                                break;
                            }
                        case "squat":
                            {
                                this.jumpAttackNum = 0
                                this.jumpTimes = 0
                                break;
                            }
                        case "jump":
                            {
                                if (this.mesh.position.y <= 0) {
                                    this.mesh.position.y = 0
                                    if (this.keyDown.squat) {
                                        this._state.section = "squat"
                                        this._state.subsection = "main"
                                        this._state.subsubsection = 0
                                    } else {
                                        this._state.section = "stand"
                                    }
                                    // this.vector.x = 0
                                    // this._state.subsubsection = 2
                                }
                                break;
                            }
                        default:
                            break;
                    }
                    break;
                }
            case "attack":
                {
                    // this.cumulativeDamage = 0

                    switch (this._state.section) {
                        case "stand":
                            {
                                this.vector.x = 0
                                break;
                            }
                        case "squat":
                            {
                                switch (this._state.subsection) {
                                    case "small":
                                        {
                                            if (this._state.subsubsection == 1) {
                                                this.vector.x /= 1.3
                                            }
                                            if (this._state.subsubsection == 2) {
                                                this.vector.x = 0
                                            }
                                            break;
                                        }
                                    default:
                                        break;
                                }
                                break;
                            }
                        case "jump":
                            {
                                if (this.mesh.position.y <= 0) {
                                    if (this._state.subsection == "fall") {

                                    } else {
                                        this._state.chapter = "normal"
                                        this._state.subsection = "main"
                                        this._state.subsubsection = 0
                                    }
                                }
                                break;
                            }
                        default:
                            break;
                    }
                    break;
                }
            case "hitRecover":
                {
                    if (this._state.section == "jump") {
                        if (this.mesh.position.y <= 0) {
                            // this._state.section = "reStand"
                            // this._state.subsection = "main"
                            this._state.subsubsection = 3
                        }
                    }
                    break;
                }
            default:
                break;
        }
        if (this._state.chapter == "normal" && this._state.section == "stand") {
            if (this.keyDown.right && this.keyDown.left) {
                this._state.subsection = "main"
            } else {
                if (this.keyDown.left) {
                    if (this.shouldFaceTo == "left") {
                        this._state.subsection = "forward"
                    } else {
                        this._state.subsection = "backward"
                    }
                } else if (this.keyDown.right) {
                    if (this.shouldFaceTo == "right") {
                        this._state.subsection = "forward"
                    } else {
                        this._state.subsection = "backward"
                    }
                }
            }
        }
        if (this._state.subsection == "forward") {
            this.vector.x = this.shouldFaceTo == "right" ? -0.1 : 0.1
        } else if (this._state.subsection == "backward") {
            this.vector.x = this.shouldFaceTo == "left" ? -0.075 : 0.075
        }
        if (this._state.chapter == "normal") {
            if (this._state.section == "squat") {
                this.vector.x *= 0.5
            }
        }
        if (this.mesh.position.y > 0) {
            if (this.isHit && this._state.subsection != "fall") {
                this.vector.y = 0
            } else {
                if (`${this._state.chapter}:${this._state.section}:${this._state.subsection}:${this._state.subsubsection}` == "hitRecover:jump:large:0") {
                    if (this.beInjuredObj.beHitVector.x == 0 && this.beInjuredObj.beHitVector.y == 0) {
                        this.vector.y = 0
                    } else {
                        this.vector.y -= 0.02

                    }
                } else if (this._state.subsection == "fall") {
                    this.vector.y -= 0.04
                    this.vector.x = 0
                } else {
                    this.vector.y -= 0.02
                }
            }
        } else {
            this.mesh.position.y = 0
            this.vector.y = 0
        }
        //Actor Intersect Collisions
        if (this.bodyBox.intersectsMesh(this.opponent.bodyBox, true)) {
            if (this.shouldFaceTo == "left") {
                this.mesh.position.x -= 0.05
            } else {
                this.mesh.position.x += 0.05
            }
            this.vector.x = 0
        }
        this.mesh.position = this.mesh.position.add(this.vector)

        if (this.mesh.position.x > 11) { this.mesh.position.x = 11 }
        if (this.mesh.position.x < -11) { this.mesh.position.x = -11 }


        if (this._state.chapter == "normal" && this._state.section != "jump") {
            if (this.shouldFaceTo == "left") {
                this._faceTo = "left"
                this.mesh.rotationQuaternion = new BABYLON.Vector3(0, 0, 0).toQuaternion()
            } else {
                this._faceTo = "right"
                this.mesh.rotationQuaternion = new BABYLON.Vector3(0, Math.PI, 0).toQuaternion()
            }
        }

        {
            this._bodyBox.position.x = this.mesh.position.x
            this._bodyBox.position.y = this.mesh.position.y
            this._bodyBox.position.z = this.mesh.position.z
            if (this._state.section == "squat") {
                this._bodyBox.scaling = new BABYLON.Vector3(1, 0.5, 1)
                this._bodyBox.position.y -= 0.2
            } else if (this._state.subsection == "fall") {
                this._bodyBox.scaling = new BABYLON.Vector3(0.2, 1.5, 0.2)
                this._bodyBox.position.y += 0.4
            } else {
                this._bodyBox.scaling = new BABYLON.Vector3(1, 1, 1)
            }
        }

        this.skeleton.bones.forEach((bone, index) => {
            this.collisionBoxes[index].PhysicsImpostor.syncImpostorWithBone(bone, this.mesh)
        })
        let attackBox = Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection][this._state.subsubsection].boxes
        if (attackBox && !this.opponent.isInvincible) {
            attackBox.forEach((boxIndex) => {
                let hitVector = Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection][this._state.subsubsection].hitVector || [0, 0, 0]
                hitVector = new BABYLON.Vector3(...hitVector)
                hitVector.x = this.shouldFaceTo == "left" ? hitVector.x : hitVector.x * -1

                let atk = Actor.actionSet()[this._state.chapter][this._state.section][this._state.subsection][this._state.subsubsection].atk
                if (!this.isHit) {
                    if (this._state.chapter == "attack") {
                        if (boxIndex != "body") {
                            if (this.collisionBoxes[boxIndex].intersectsMesh(this.opponent.bodyBox, true)) {
                                this.opponent.setBeInjuredObj(atk, this._state.subsection, hitVector)
                                this.isHit = true
                            }
                        } else {
                            if (this.bodyBox.intersectsMesh(this.opponent.bodyBox, true)) {
                                this.opponent.setBeInjuredObj(atk, this._state.subsection, hitVector)
                                this.isHit = true
                            }
                        }
                    }
                }
            })
        }

        if (this.beInjuredObj.atk != null) {
            this.beInjured()
        }

        // this.collisionBoxes.forEach((thisBox) => {
        //     this.opponent.collisionBoxes.forEach((oppoBox) => {
        //         if (thisBox.intersectsMesh(oppoBox, true)) {
        //             if (this._state.chapter == "attack") {

        //                 console.log("c")
        //             }
        //         }
        //     })
        // })

    }

    setOpponent(opponent) {
        this._opponent = opponent
    }
    get opponent() {
        return this._opponent
    }
    get shouldFaceTo() {
        return this.opponent.mesh.position.x > this.mesh.position.x ? "left" : "right"
    }
    get isInvincible() {
        return ((this._state.section == "reStand") || (`${this._state.chapter}:${this._state.subsection}` == "hitRecover:large" && this._state.subsubsection >= 1)) && (this.mesh.position.y <= 0)
    }

    setBeInjuredObj(atk = 100, scale = "small", beHitVector = BABYLON.Vector3.Zero()) {
        // beHitVector.x = this.shouldFaceTo == "left" ? beHitVector.x * -1 : beHitVector.x
        let isBack = () => {
            return (this.keyDown.left && this._faceTo == "right") || (this.keyDown.right && this._faceTo == "left")
        }
        if (this.keyDown.left != this.keyDown.right && isBack() && (this._state.chapter == "normal" || this._state.chapter == "defense")) {
            this._state.chapter = "defense"
            this._state.subsection = "main"
            this._state.subsubsection = 0
            if (this.perfectDefenseTime >= 0) {
                this.isPD = true

                let tempM = new BABYLON.StandardMaterial("material", this.scene)
                tempM.diffuseColor = new BABYLON.Color3(1, 1, 1)
                tempM.specularColor = new BABYLON.Color3(1, 1, 1)
                tempM.emissiveColor = new BABYLON.Color3(1, 1, 1)
                tempM.ambientColor = new BABYLON.Color3(1, 1, 1)

                this.materialMesh.material = tempM
                // console.log("perfect")

                this.opponent.cumulativeDamage = this.cumulativeDamage * (2 / 3)
                this.cumulativeDamage /= 3
            } else {
                this.HP -= atk / 5
                this.cumulativeDamage += atk / 5
            }
        } else {
            this.beInjuredObj = { atk: atk, scale: scale == "fall" ? "small" : scale, beHitVector: beHitVector }
        }
        if (beHitVector.x == 0) {
            this.mesh.position.x += this.shouldFaceTo == "left" ? -0.2 : 0.2
        }
        // if (this._state.subsection != "backward") {
        //     this.beInjuredObj = { atk: atk, scale: scale, beHitVector: beHitVector }
        // } else {
        //     this._state.chapter = "defense"
        //     this._state.subsection = "main"
        //     this._state.subsubsection = 0
        // }
    }

    beInjured() {
        if (!(this._state.chapter == "attack" && this._state.subsection == "large" && this._state.subsubsection <= 1) || (this.cumulativeDamage >= this.maxCumulativeDamage)) {
            if (this.beInjuredObj.beHitVector.y > 0) {
                this._state.section = "jump"
            }
            this._state.chapter = "hitRecover"
            this._state.subsection = this._state.section == "jump" ? "large" : this.beInjuredObj.scale
            this._state.subsubsection = 0
            this.vector = this.beInjuredObj.beHitVector
            this.mesh.position = this.mesh.position.add(this.beInjuredObj.beHitVector)
            this.isHit = false
        }
        this.beHitNum += 1
        this.HP -= this.beInjuredObj.atk / this.beHitNum
        // console.log(this.HP)
        // console.log(this.vector) 

        if (this.cumulativeDamage >= this.maxCumulativeDamage) {
            this.HP -= this.cumulativeDamage
            this.cumulativeDamage = 0
        } else {
            this.cumulativeDamage += this.beInjuredObj.atk / this.beHitNum
        }

        this.beInjuredObj.atk = null
        // this.isHit = false
    }

    restart() {
        this.jumpAttackNum = 0
        this.isHit = false
        this.jumpTimes = 0
        this._state.chapter = "normal"
        this._state.section = "stand"
        this._state.subsection = "main"
        this._state.subsubsection = 0
        this.mesh.position = this._startPosition.clone()
        this.mesh.rotationQuaternion = this._startRotationQuaternion.clone()
        this.vector = new BABYLON.Vector3.Zero()
        this.isPD = false
        this.perfectDefenseTime = -1
        this.HP = this.maxHP
        this.beHitNum = 0
        this.beInjuredObj = { atk: null, scale: null, beHitVector: BABYLON.Vector3.Zero() }
        this.cumulativeDamage = 0
    }
}