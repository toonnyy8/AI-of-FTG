import { Game } from "../../lib/slime-FTG-for-cnn/src/js/game"

const keySets: [
    {
        jump: string;
        squat: string;
        left: string;
        right: string;
        attack: {
            light: string;
            medium: string;
            heavy: string;
        };
    },
    {
        jump: string;
        squat: string;
        left: string;
        right: string;
        attack: {
            light: string;
            medium: string;
            heavy: string;
        };
    }
] = [
        {
            jump: "w",
            squat: "s",
            left: "a",
            right: "d",
            attack: {
                light: "j",
                medium: "k",
                heavy: "l",
            },
        },
        {
            jump: "ArrowUp",
            squat: "ArrowDown",
            left: "ArrowLeft",
            right: "ArrowRight",
            attack: {
                light: "1",
                medium: "2",
                heavy: "3",
            },
        },
    ]

let canvas = <HTMLCanvasElement>document.getElementById("bobylonCanvas")
Game(
    keySets,
    canvas,
).then(({ next, getP1, getP2, getRestart }) => {
    const loop = () => {
        next()
        requestAnimationFrame(loop)
    }
    requestAnimationFrame(loop)
})

