import * as tf from "@tensorflow/tfjs"

declare class SequenceTidy {
    funcs: [()=>{}]
    constructor(func: ()=>{})
    next(func: ()=>{}):SequenceTidy
    run(input:any):tf.Tensor 
}

export declare function sequenceTidy (func: ()=>{}):SequenceTidy

declare class MemoryManagement {
    _mem:tf.Tensor
    ptr :tf.Tensor
    constructor()
}

declare class MemoryBox{
    box:{name: MemoryManagement}
    constructor()
    read(name: String):MemoryManagement
    dispose():void
}

export declare function memoryBox():MemoryBox