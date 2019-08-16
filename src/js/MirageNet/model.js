import * as tf from "@tensorflow/tfjs"
import * as a3c from "./a3c"
import * as dddqn from "./dddqn"
import { transformerXL } from "./transformerXL"

//Forecast action reward status
export class MirageNet {
    constructor({ predictionNum = 3, stepNum = 3 }) {
        super()
        this.actionModels = [new tf.Model()]
        this.actionModels = new Array(predictionNum).fill(this.buildActionModel())
        this.predictionNet = transformer_xl()
    }

    buildActionModel() {

        return new tf.Model()
    }

    inference() {
        let actions = this.actionModels.map((m) => {
            return m.apply()
        })

        /*
         * let rewards_ = actions.map((a)=>{this.predictionNet.model.apply(a)})
         * 
         * 
         * 
         */
    }

    loss() {

    }

    train() {

    }

}