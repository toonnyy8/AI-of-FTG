import * as tf from "@tensorflow/tfjs"

declare class SequenceTidy {
    funcs: [()=>{}]
    constructor(func: ()=>{})
    next(func: ()=>{}):SequenceTidy
    run(input:any):tf.Tensor 
}

export declare function sequenceTidy (func: ()=>{}):SequenceTidy