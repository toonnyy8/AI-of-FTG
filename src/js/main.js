import "@babel/polyfill"

import * as BABYLON from './babylon-module'

import * as slime from "../../file/slime/slime.js"
import * as CANNON from "cannon"

global.CANNON = CANNON

// Get the canvas DOM element
let canvas = document.getElementById('bobylonCanvas')
// Load the 3D engine
let engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true })
// CreateScene function that creates and return the scene

function createScene() {

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
    HPBar.p1.setPivotMatrix(new BABYLON.Matrix.Translation(-3, 0, 0), false);
    HPBar.p1.position.y = 9
    HPBar.p1.position.x = 9
    HPBar.p1.material = hpMaterial


    HPBar.p2.setPivotMatrix(new BABYLON.Matrix.Translation(3, 0, 0), false);
    HPBar.p2.position.y = 9
    HPBar.p2.position.x = -9
    HPBar.p2.material = hpMaterial

    BABYLON.SceneLoader.OnPluginActivatedObservable.addOnce(loader => {
        loader.animationStartMode = BABYLON.GLTFLoaderAnimationStartMode.NONE
    })
    let player1, player2
    BABYLON.SceneLoader.ImportMesh("", slime.Actor.url(), "", scene, (meshes, particleSystems, skeletons, animationGroups) => {
        console.log(skeletons)
        console.log(meshes)
        console.log(animationGroups)

        let animationGroup = animationGroups[0] //.start(false)
        animationGroup.stop()
        player1 = new slime.Actor({
            mesh: meshes[0],
            materialMesh: meshes[1],
            animationGroup: animationGroups[0],
            skeleton: skeletons[0],
            scene: scene,
            startPosition: new BABYLON.Vector3(5, 0, 0),
            startRotationQuaternion: new BABYLON.Vector3(0, Math.PI, 0).toQuaternion()
        })

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
            player2 = new slime.Actor({
                mesh: meshes[0],
                materialMesh: meshes[1],
                animationGroup: animationGroups[0],
                skeleton: skeletons[0],
                keySet: { jump: "ArrowUp", squat: "ArrowDown", left: "ArrowLeft", right: "ArrowRight", attack: { small: "1", medium: "2", large: "3" } },
                startPosition: new BABYLON.Vector3(-5, 0, 0),
                startRotationQuaternion: new BABYLON.Vector3(0, 0, 0).toQuaternion()
            })

            player1.setOpponent(player2)
            player2.setOpponent(player1)

            engine.runRenderLoop(() => {
                player1.tick(false)
                player2.tick(false)
                scene.render()
                if (player1.isHit || player2.isHit) {
                    camera.setTarget(new BABYLON.Vector3(-0.06 + Math.random() * 0.12, 3.94 + Math.random() * 0.12, 0));
                } else {
                    camera.setTarget(new BABYLON.Vector3(0, 4, 0));
                }
                HPBar.p1.scaling.x = player1.HP / 3000
                HPBar.p2.scaling.x = player2.HP / 3000
                if (player1.HP <= 0 || player2.HP <= 0) {
                    player1.restart()
                    player2.restart()
                }
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