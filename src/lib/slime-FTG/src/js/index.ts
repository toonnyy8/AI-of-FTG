import * as slime from "../../file/slime/slime.js"

export declare class Game {
    player1: slime.Actor
    player2: slime.Actor
    restart: Boolean
    constructor(keySets: [{}, {}], canvas: HTMLElement)
}
export * from "./game.js"
