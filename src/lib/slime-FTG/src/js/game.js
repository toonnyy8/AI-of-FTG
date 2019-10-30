import "core-js/stable"
import "regenerator-runtime/runtime"

import * as BABYLON from './babylon-module'

import * as slime from "../../file/slime/slime.js"
import * as CANNON from "cannon"

global.CANNON = CANNON

export class Game {
    constructor(keySets) {

        // Get the canvas DOM element
        canvas = document.getElementById('bobylonCanvas')
        // Load the 3D engine
        engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true })
        // CreateScene function that creates and return the scene

        this.player1
        this.player2
        this.restart = true

        let createScene = () => {

            // This creates a basic Babylon Scene object (non-mesh)
            let scene = new BABYLON.Scene(engine)
            scene.enablePhysics()

            //Adding an Arc Rotate Camera
            // let camera = new BABYLON.ArcRotateCamera("Camera", Math.PI / 2, Math.PI / 2, 15, new BABYLON.Vector3(0, 5, 0), scene)
            var camera = new BABYLON.FreeCamera("camera1", new BABYLON.Vector3(0, 10, 100), scene);
            camera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;

            camera.orthoTop = 9 * 0.7;
            camera.orthoBottom = -9 * 0.7;
            camera.orthoLeft = -16 * 0.7;
            camera.orthoRight = 16 * 0.7;
            // target the camera to scene origin
            camera.setTarget(new BABYLON.Vector3(0, 4, 0));
            // attach the camera to the canvas
            // camera.attachControl(canvas, false)
            //Adding a light
            let light = new BABYLON.HemisphericLight('light1', new BABYLON.Vector3(0, 5, 0), scene);
            light.diffuse = new BABYLON.Color3(0.9, 0.85, 0.8);
            light.specular = new BABYLON.Color3(1, 1, 1);
            light.groundColor = new BABYLON.Color3(0.4, 0.4, 0.5);

            let hpMaterial = new BABYLON.StandardMaterial("hpMaterial", scene);
            hpMaterial.diffuseColor = new BABYLON.Color3(1, 0.3, 0.3)
            hpMaterial.specularColor = new BABYLON.Color3(1, 0.3, 0.3)
            hpMaterial.emissiveColor = new BABYLON.Color3(1, 0.3, 0.3)
            hpMaterial.ambientColor = new BABYLON.Color3(1, 0.3, 0.3)

            let HPBar = { p1: BABYLON.MeshBuilder.CreateBox("P1_hp", { size: 0.5, width: 8 }), p2: BABYLON.MeshBuilder.CreateBox("P2_hp", { size: 0.5, width: 8 }) }
            HPBar.p1.setPivotMatrix(new BABYLON.Matrix.Translation(-4, 0, 0), false);
            HPBar.p1.position.y = 9
            HPBar.p1.position.x = 10
            HPBar.p1.material = hpMaterial


            HPBar.p2.setPivotMatrix(new BABYLON.Matrix.Translation(4, 0, 0), false);
            HPBar.p2.position.y = 9
            HPBar.p2.position.x = -10
            HPBar.p2.material = hpMaterial

            let cumulativeDamageMaterial = new BABYLON.StandardMaterial("hpMaterial", scene);
            cumulativeDamageMaterial.diffuseColor = new BABYLON.Color3(0.5, 0.8, 0.5)
            cumulativeDamageMaterial.specularColor = new BABYLON.Color3(0.5, 0.8, 0.5)
            cumulativeDamageMaterial.emissiveColor = new BABYLON.Color3(0.5, 0.8, 0.5)
            cumulativeDamageMaterial.ambientColor = new BABYLON.Color3(0.5, 0.8, 0.5)

            let cumulativeDamageBar = { p1: BABYLON.MeshBuilder.CreateBox("P1_ cumulativeDamage", { size: 0.1, width: 8 }), p2: BABYLON.MeshBuilder.CreateBox("P2_cumulativeDamage", { size: 0.1, width: 8 }) }
            cumulativeDamageBar.p1.setPivotMatrix(new BABYLON.Matrix.Translation(-4, 0, 0), false);
            cumulativeDamageBar.p1.position.y = 8.5
            cumulativeDamageBar.p1.position.x = 10
            cumulativeDamageBar.p1.material = cumulativeDamageMaterial


            cumulativeDamageBar.p2.setPivotMatrix(new BABYLON.Matrix.Translation(4, 0, 0), false);
            cumulativeDamageBar.p2.position.y = 8.5
            cumulativeDamageBar.p2.position.x = -10
            cumulativeDamageBar.p2.material = cumulativeDamageMaterial

            BABYLON.SceneLoader.OnPluginActivatedObservable.addOnce(loader => {
                loader.animationStartMode = BABYLON.GLTFLoaderAnimationStartMode.NONE
            })
            BABYLON.SceneLoader.ImportMesh("", slime.Actor.url(), "", scene, (meshes, particleSystems, skeletons, animationGroups) => {
                console.log(skeletons)
                console.log(meshes)
                console.log(animationGroups)

                let animationGroup = animationGroups[0] //.start(false)
                animationGroup.stop()
                this.player1 = new slime.Actor({
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
                    maxPerfectDefenseTime: 10
                })

                // var skeletonViewer = new BABYLON.Debug.SkeletonViewer(skeletons[0], meshes[0], scene);// Create a skeleton viewer for the mesh
                // skeletonViewer.isEnabled = true; // Enable it
                // skeletonViewer.color = BABYLON.Color3.Red(); // Change default color from white to red

                BABYLON.SceneLoader.ImportMesh("", slime.Actor.url(), "", scene, (meshes, particleSystems, skeletons, animationGroups) => {
                    console.log(skeletons)
                    console.log(meshes)
                    console.log(animationGroups)

                    meshes[1].material.diffuseColor = new BABYLON.Color3(0.1, 0, 0);
                    meshes[1].material.specularColor = new BABYLON.Color3(0.5, 0, 0);
                    meshes[1].material.emissiveColor = new BABYLON.Color3(0.5, 0, 0);
                    meshes[1].material.ambientColor = new BABYLON.Color3(0.5, 0, 0);

                    let animationGroup = animationGroups[0]
                    animationGroup.stop()
                    this.player2 = new slime.Actor({
                        mesh: meshes[0],
                        materialMesh: meshes[1],
                        animationGroup: animationGroups[0],
                        skeleton: skeletons[0],
                        keySet: keySets[1],
                        // keySet: { jump: "ArrowUp", squat: "ArrowDown", left: "ArrowLeft", right: "ArrowRight", attack: { small: "1", medium: "2", large: "3" } },
                        startPosition: new BABYLON.Vector3(-5, 0, 0),
                        startRotationQuaternion: new BABYLON.Vector3(0, 0, 0).toQuaternion(),
                        maxHP: 3000,
                        maxCumulativeDamage: 500,
                        maxPerfectDefenseTime: 10
                    })

                    this.player1.setOpponent(this.player2)
                    this.player2.setOpponent(this.player1)

                    engine.runRenderLoop(() => {
                        this.player1.tick(false)
                        this.player2.tick(false)
                        scene.render()
                        if (this.player1.isHit || this.player2.isHit) {
                            camera.setTarget(new BABYLON.Vector3(-0.06 + Math.random() * 0.12, 3.94 + Math.random() * 0.12, 0));
                        } else {
                            camera.setTarget(new BABYLON.Vector3(0, 4, 0));
                        }

                        if (this.player1.HP > this.player1.maxHP) {
                            this.player1.HP = this.player1.maxHP
                        }
                        if (this.player2.HP > this.player2.maxHP) {
                            this.player2.HP = this.player2.maxHP
                        }
                        HPBar.p1.scaling.x = this.player1.HP / this.player1.maxHP
                        HPBar.p2.scaling.x = this.player2.HP / this.player2.maxHP
                        if (this.player1.HP <= 0 || this.player2.HP <= 0) {
                            if (this.player1.HP > 0) {
                                console.log("Player1 win")
                            } else if (this.player2.HP > 0) {
                                console.log("Player2 win")
                            } else {
                                console.log("Draw")
                            }
                            this.player1.restart()
                            this.player2.restart()
                            this.restart = true
                        }

                        if (this.player1.cumulativeDamage > this.player1.maxCumulativeDamage) {
                            this.player1.cumulativeDamage = this.player1.maxCumulativeDamage
                        }
                        if (this.player2.cumulativeDamage > this.player2.maxCumulativeDamage) {
                            this.player2.cumulativeDamage = this.player2.maxCumulativeDamage
                        }
                        cumulativeDamageBar.p1.scaling.x = this.player1.cumulativeDamage / this.player1.maxCumulativeDamage
                        cumulativeDamageBar.p2.scaling.x = this.player2.cumulativeDamage / this.player2.maxCumulativeDamage


                    })

                }, null, null, ".glb")
            }, null, null, ".glb")

            // create a built-in "ground" shape;
            var ground = BABYLON.Mesh.CreateGround('ground1', 60, 60, 2, scene);
            ground.position.y = -0.1
            ground.position.z = 20
            return scene

        }
        // call the createScene function
        let scene = createScene()
        // run the render loop
        // engine.runRenderLoop(() => {
        //     scene.render()
        // })
        // the canvas/window resize event handler
        window.addEventListener('resize', () => {
            engine.resize()
        })

        window.onresize = (e) => {
            if (document.body.offsetWidth / document.body.offsetHeight > 1920 / 1080) {
                canvas.style = "height:100%"
            } else {
                canvas.style = "width:100%"
            }
        }
        if (document.body.offsetWidth / document.body.offsetHeight > 1920 / 1080) {
            canvas.style = "height:100%"
        } else {
            canvas.style = "width:100%"
        }
    }
}