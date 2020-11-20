import "core-js/stable"
import "regenerator-runtime/runtime"

import * as BABYLON from "./babylon-module"

import * as slime from "../../file/slime/slime.js"
import * as CANNON from "cannon"

global["CANNON"] = CANNON

export const Game = (
    keySets: [
        {
            jump: string
            squat: string
            left: string
            right: string
            attack: {
                light: string
                medium: string
                heavy: string
            }
        },
        {
            jump: string
            squat: string
            left: string
            right: string
            attack: {
                light: string
                medium: string
                heavy: string
            }
        }
    ],
    canvas: HTMLCanvasElement
) => {
    // Get the canvas DOM element
    // canvas = document.getElementById('bobylonCanvas')
    // Load the 3D engine
    const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true })
    // window.addEventListener('resize', () => {
    //     engine.resize()
    // })

    window.onresize = (e) => {
        if (document.body.offsetWidth / document.body.offsetHeight > 1080 / 1080) {
            canvas.style.height = "100%"
            canvas.style.width = "auto"
        } else {
            canvas.style.height = "auto"
            canvas.style.width = "100%"
        }
    }
    if (document.body.offsetWidth / document.body.offsetHeight > 1080 / 1080) {
        canvas.style.height = "100%"
        canvas.style.width = "auto"
    } else {
        canvas.style.height = "auto"
        canvas.style.width = "100%"
    }

    // CreateScene function that creates and return the scene

    let player1
    let player2
    let restart = true

    // This creates a basic Babylon Scene object (non-mesh)
    let scene = new BABYLON.Scene(engine)
    scene.clearColor = new BABYLON.Color4(0, 0, 0)
    scene.enablePhysics(new BABYLON.Vector3(0, 1, 0))

    //Adding an Arc Rotate Camera
    // let camera = new BABYLON.ArcRotateCamera("Camera", Math.PI / 2, Math.PI / 2, 15, new BABYLON.Vector3(0, 5, 0), scene)
    var camera = new BABYLON.FreeCamera("camera1", new BABYLON.Vector3(0, 10, 100), scene)
    camera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA

    camera.orthoTop = 11 * 0.7
    camera.orthoBottom = -9 * 0.7
    camera.orthoLeft = -10 * 0.7
    camera.orthoRight = 10 * 0.7
    // target the camera to scene origin
    camera.setTarget(new BABYLON.Vector3(0, 4, 0))
    // attach the camera to the canvas
    // camera.attachControl(canvas, false)
    //Adding a light
    const createLight = (name = "", color = { r: 0.2, g: 0.2, b: 0.2 }) => {
        const l = new BABYLON.HemisphericLight(name, new BABYLON.Vector3(0, 5, 0), scene)
        l.diffuse = new BABYLON.Color3(color.r, color.g, color.b)
        l.specular = new BABYLON.Color3(color.r, color.g, color.b)
        l.groundColor = new BABYLON.Color3(color.r, color.g, color.b)
        return l
    }
    let light = createLight("light")

    const createMaterial = (name = "", color = { r: 0.4, g: 0.4, b: 1 }) => {
        const m = new BABYLON.StandardMaterial(name, scene)
        m.diffuseColor = new BABYLON.Color3(color.r, color.g, color.b)
        m.specularColor = new BABYLON.Color3(color.r, color.g, color.b)
        m.emissiveColor = new BABYLON.Color3(color.r, color.g, color.b)
        m.ambientColor = new BABYLON.Color3(color.r, color.g, color.b)
        return m
    }

    const createBar = (
        name: string,
        size: number,
        width: number,
        pos: {
            x: number
            y: number
        },
        center: {
            x: number
            y: number
            z: number
        },
        material: BABYLON.Material
    ) => {
        const bar = BABYLON.MeshBuilder.CreateBox(name, { size, width })
        bar.setPivotMatrix(BABYLON.Matrix.Translation(center.x, center.y, center.z), false)
        bar.position.y = pos.y
        bar.position.x = pos.x
        bar.material = material
        return bar
    }
    let HPBar = {
        p1: createBar(
            "P1_hp",
            0.5,
            6,
            { x: 6.5, y: 11 },
            { x: -3, y: 0, z: 0 },
            createMaterial("p1_hpMaterial", { r: 0.4, g: 0.4, b: 1 })
        ),
        p2: createBar(
            "P2_hp",
            0.5,
            6,
            { x: -6.5, y: 11 },
            { x: 3, y: 0, z: 0 },
            createMaterial("p2_hpMaterial", { r: 1, g: 0.4, b: 0.4 })
        ),
    }

    let cumulativeDamageBar = {
        p1: createBar(
            "P1_cumulativeDamage",
            0.2,
            6,
            { x: 6.5, y: 10 },
            { x: -3, y: 0, z: 0 },
            createMaterial("p1_cumulativeDamageMaterial", { r: 0.4, g: 0.4, b: 1 })
        ),
        p2: createBar(
            "P2_cumulativeDamage",
            0.2,
            6,
            { x: -6.5, y: 10 },
            { x: 3, y: 0, z: 0 },
            createMaterial("p2_cumulativeDamageMaterial", { r: 1, g: 0.4, b: 0.4 })
        ),
    }

    return BABYLON.SceneLoader.ImportMeshAsync("", slime.Actor.url(), "", scene, null, ".glb")
        .then(({ meshes, particleSystems, skeletons, animationGroups }) => {
            console.log(skeletons)
            console.log(meshes)
            console.log(animationGroups)

            meshes[1].material = createMaterial("p1_material", { r: 0.4, g: 0.4, b: 1 })

            let animationGroup = animationGroups[0] //.start(false)
            animationGroup.stop()
            player1 = new slime.Actor({
                mesh: meshes[0],
                materialMesh: meshes[1],
                animationGroup: animationGroups[0],
                skeleton: skeletons[0],
                keySet: keySets[0],
                scene: scene,
                startPosition: new BABYLON.Vector3(5, 0, 0),
                startRotationQuaternion: new BABYLON.Vector3(0, Math.PI, 0).toQuaternion(),
                maxHP: 3000,
                maxCumulativeDamage: 500,
                maxPerfectDefenseTime: 10,
            })
            return BABYLON.SceneLoader.ImportMeshAsync("", slime.Actor.url(), "", scene, null, ".glb")
        })
        .then(({ meshes, particleSystems, skeletons, animationGroups }) => {
            console.log(skeletons)
            console.log(meshes)
            console.log(animationGroups)
            meshes[1].material = createMaterial("p2_material", { r: 1, g: 0.4, b: 0.4 })

            let animationGroup = animationGroups[0]
            animationGroup.stop()
            player2 = new slime.Actor({
                mesh: meshes[0],
                materialMesh: meshes[1],
                animationGroup: animationGroups[0],
                skeleton: skeletons[0],
                keySet: keySets[1],
                scene: scene,
                startPosition: new BABYLON.Vector3(-5, 0, 0),
                startRotationQuaternion: new BABYLON.Vector3(0, 0, 0).toQuaternion(),
                maxHP: 3000,
                maxCumulativeDamage: 500,
                maxPerfectDefenseTime: 10,
            })
        })
        .then(() => {
            // create a built-in "ground" shape;
            var ground = BABYLON.Mesh.CreateGround("ground1", 60, 60, 2, scene)
            ground.position.y = -0.1
            ground.position.z = 20

            player1.setOpponent(player2)
            player2.setOpponent(player1)

            const next = () => {
                restart = false
                player1.tick(false)
                player2.tick(false)

                if (player1.isHit || player2.isHit) {
                    camera.setTarget(new BABYLON.Vector3(-0.06 + Math.random() * 0.12, 3.94 + Math.random() * 0.12, 0))
                } else {
                    camera.setTarget(new BABYLON.Vector3(0, 4, 0))
                }

                if (player1.HP > player1.maxHP) {
                    player1.HP = player1.maxHP
                }
                if (player2.HP > player2.maxHP) {
                    player2.HP = player2.maxHP
                }
                HPBar.p1.scaling.x = player1.HP / player1.maxHP
                HPBar.p2.scaling.x = player2.HP / player2.maxHP
                if (player1.HP <= 0 || player2.HP <= 0) {
                    if (player1.HP > 0) {
                        console.log("Player1 win")
                    } else if (player2.HP > 0) {
                        console.log("Player2 win")
                    } else {
                        console.log("Draw")
                    }
                    player1.restart()
                    player2.restart()
                    restart = true
                }

                if (player1.cumulativeDamage > player1.maxCumulativeDamage) {
                    player1.cumulativeDamage = player1.maxCumulativeDamage
                }
                if (player2.cumulativeDamage > player2.maxCumulativeDamage) {
                    player2.cumulativeDamage = player2.maxCumulativeDamage
                }
                cumulativeDamageBar.p1.scaling.x = player1.cumulativeDamage / player1.maxCumulativeDamage
                cumulativeDamageBar.p2.scaling.x = player2.cumulativeDamage / player2.maxCumulativeDamage
            }
            const render = () => {
                scene.render()
            }
            // engine.runRenderLoop(next)

            return { next, render, getP1: () => player1, getP2: () => player2, getRestart: () => restart, scene }
        })
}
